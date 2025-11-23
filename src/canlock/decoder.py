from uuid import UUID

import pandas as pd
from sqlmodel import Session as DbSession
from sqlmodel import select
from tqdm import tqdm

from canlock.db.database import get_session, init_db
from canlock.db.models import (
    AnalogAttributes,
    CanMessage,
    DefinedDigitalValues,
    PgnDefinition,
    Session,
    SpnDefinition,
)


class SessionDecoder:
    """
    Decodes CAN messages for a specific session.

    This class handles fetching CAN messages from the database, identifying
    Parameter Group Numbers (PGNs), and extracting Suspect Parameter Numbers (SPNs)
    values based on defined rules.
    """
    def __init__(self, db: DbSession):
        """
        Initialize the SessionDecoder.

        Args:
            db (DbSession): The database session to use for queries.
        """
        self.db = db

    @staticmethod
    def list_sessions(print_all: bool = False) -> list[Session]:
        """
        List the first 5 sessions in the database.
        """
        init_db()
        with get_session() as session:
            sessions = session.exec(select(Session)).all()

            if print_all:
                for s in sessions:
                    print(f"Session ID: {s.id}, Start: {s.start_time}, End: {s.end_time}")
        
        return sessions

    @staticmethod
    def extract_pgn_number_from_payload(identifier: int) -> int:
        """
        Extract the PGN (Parameter Group Number) from the CAN identifier.

        Args:
            identifier (int): The CAN identifier.

        Returns:
            int: The extracted PGN.
        """
        binary_identifier = bin(identifier)[2:].zfill(29)
        pgn_identifier = binary_identifier[3:21]
        pgn_integer = int(pgn_identifier, 2)
        
        return pgn_integer

    @staticmethod
    def extract_spn_bits_from_payload(spn: SpnDefinition, payload: bytes) -> int:
        """
        Extract the raw bits for an SPN from the payload.

        Args:
            spn (SpnDefinition): The SPN definition containing bit start and length.
            payload (bytes): The raw CAN message payload.

        Returns:
            int: The extracted raw integer value.
        """
        int_payload = int(payload.hex(), 16)
        binary_payload = bin(int_payload)[2:].zfill(64)
        bit_start = spn.bit_start
        bit_length = spn.bit_length
        
        return int(binary_payload[bit_start:bit_start+bit_length], 2)

    def extract_values_from_spns(self, spn_list: list[SpnDefinition], analogic_rules: list[AnalogAttributes], payload: bytes) -> dict[str, float | str]:
        """
        Extract and scale values for a list of SPNs from the payload.

        Args:
            spn_list (list[SpnDefinition]): List of SPNs to extract.
            analogic_rules (list[AnalogAttributes]): List of analog attributes for scaling.
            payload (bytes): The raw CAN message payload.

        Returns:
            dict[str, float | str]: A dictionary mapping SPN names to their scaled or decoded values.
        """
        spn_values = {}
        for spn, analog_attr in zip(spn_list, analogic_rules):
            spn_pre_val = self.extract_spn_bits_from_payload(spn, payload)
            
            if spn.is_analog:
                if analog_attr is None:
                    continue
                spn_val = analog_attr.scale * spn_pre_val + analog_attr.offset
                spn_values[spn.name] = spn_val
            else:
                spn_val = self.decode_digital_value(spn.id, spn_pre_val)
                if spn_val is not None:
                    spn_values[spn.name] = spn_val
        return spn_values

    def decode_digital_value(self, spn_id: UUID, raw_value: int) -> str | None:
        """
        Decode a digital value to its categorical description.

        Args:
            spn_id (UUID): The ID of the SPN.
            raw_value (int): The raw integer value extracted from the payload.

        Returns:
            str | None: The categorical description if found, otherwise None.
        """
        query = select(DefinedDigitalValues).where(
            DefinedDigitalValues.spn_id == spn_id,
            DefinedDigitalValues.value == raw_value
        )
        result = self.db.exec(query).first()
        
        if result:
            return result.name
        return None

    def decode(self, session_id: UUID) -> pd.DataFrame:
        """
        Decode all messages for a given session.

        Args:
            session_id (UUID): The ID of the session to decode.

        Returns:
            pd.DataFrame: A DataFrame containing the decoded data, with timestamps and SPN values.
        """
        can_messages = self.db.exec(
            select(CanMessage)
            .where(CanMessage.session_id == session_id)
            .order_by(CanMessage.timestamp)
        ).all()

        decoded_data = []
        pgn_cache = {}
        for message in tqdm(can_messages, desc="Decoding messages"):
            if message.can_identifier is None or message.payload is None:
                continue

            pgn_number = self.extract_pgn_number_from_payload(message.can_identifier)
            
            if pgn_number not in pgn_cache:
                pgn_def = self.db.exec(
                    select(PgnDefinition).where(PgnDefinition.pgn_identifier == pgn_number)
                ).first()
                
                if pgn_def:
                    spns = self.db.exec(
                        select(SpnDefinition).where(SpnDefinition.pgn_id == pgn_def.id)
                    ).all()
                    
                    spn_rules = []
                    for spn in spns:
                        spn_rules.append((spn, spn.analog_attributes))
                    
                    pgn_cache[pgn_number] = spn_rules
                else:
                    pgn_cache[pgn_number] = None

            spn_rules = pgn_cache[pgn_number]
            
            if spn_rules:
                spns = [s[0] for s in spn_rules]
                analog_attrs = [s[1] for s in spn_rules]
                
                values = self.extract_values_from_spns(spns, analog_attrs, message.payload)
                if values:
                    row = {"timestamp": message.timestamp}
                    row.update(values)
                    decoded_data.append(row)

        if not decoded_data:
            return pd.DataFrame()

        return pd.DataFrame(decoded_data)
