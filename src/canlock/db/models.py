import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import BigInteger
from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint
from sqlalchemy import Column


class VehicleBase(SQLModel):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    license_plate: str
    model: str
    make: str
    year: int


class Vehicle(VehicleBase, table=True):
    """Represents a Vehicle."""
    sessions: list["Session"] = Relationship(back_populates="vehicle")
    pgn_definitions: list["PgnDefinition"] = Relationship(back_populates="vehicle")
    ecus: list["Ecu"] = Relationship(back_populates="vehicle")

    __table_args__ = (
        UniqueConstraint(
            "license_plate",
            "make",
            "year",
            name="uq_vehicle_fields",
        ),
    )


class LogFile(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    hash: str
    session_id: UUID | None = Field(foreign_key="session.id")
    session: Optional["Session"] = Relationship(back_populates="log_files")

class SessionBase(SQLModel):
    """Represents a Session."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    start_time: datetime.datetime
    end_time: datetime.datetime | None
    vehicle_id: UUID = Field(foreign_key="vehicle.id")


class Session(SessionBase, table=True):
    """Represents a Session."""

    vehicle: Optional["Vehicle"] = Relationship(back_populates="sessions")
    log_files: list["LogFile"] = Relationship(back_populates="session", cascade_delete=True)
    can_messages: list["CanMessage"] = Relationship(back_populates="session", cascade_delete=True)


class CanMessageBase(SQLModel):
    timestamp: datetime.datetime
    can_identifier: int | None

    length: int | None
    session_id: UUID | None = Field(foreign_key="session.id")


class CanMessage(CanMessageBase, table=True):
    """Represents a CAN Message."""

    id: int | None = Field(default=None, primary_key=True)
    payload: bytes | None
    session: Optional["Session"] = Relationship(back_populates="can_messages")


class PgnDefinitionBase(SQLModel):
    """Represents a PGN response."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    pgn_identifier: int
    name: str | None = None


class PgnDefinition(PgnDefinitionBase, table=True):
    """Represents a PGN Definition."""
    spns: list["SpnDefinition"] = Relationship(back_populates="pgn")
    vehicle_id: UUID | None = Field(default=None, foreign_key="vehicle.id")
    vehicle: Optional["Vehicle"] = Relationship(back_populates="pgn_definitions")


class SpnDefinitionBase(SQLModel):
    """Represents a SPN response."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    spn_identifier: int | None = None
    name: str | None = None
    is_analog: bool
    bit_start: int
    bit_length: int


class SpnDefinition(SpnDefinitionBase, table=True):
    """Represents an SPN Definition."""

    pgn_id: UUID | None = Field(default=None, foreign_key="pgndefinition.id")
    analog_attributes_id: UUID | None = Field(
        default=None, foreign_key="analogattributes.id"
    )
    pgn: PgnDefinition | None = Relationship(back_populates="spns")
    analog_attributes: Optional["AnalogAttributes"] = Relationship(
        back_populates="spn", sa_relationship_kwargs={"uselist": False}
    )
    digital_values: list["DefinedDigitalValues"] = Relationship(back_populates="spn")


class EcuBase(SQLModel):
    """Represents an EcuBase."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    address: int = Field(default=None)
    name: str


class Ecu(EcuBase, table=True):
    """Represents an ECU."""
    vehicle_id: UUID | None = Field(default=None, foreign_key="vehicle.id")
    vehicle: Optional["Vehicle"] = Relationship(back_populates="ecus")



class DefinedDigitalValuesBase(SQLModel):
    """Represents a Defined Digital Values Base."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    value: int  = Field(sa_column=Column(BigInteger))


class DefinedDigitalValues(DefinedDigitalValuesBase, table=True):
    """Represents Defined Digital Values."""

    spn_id: UUID = Field(
        foreign_key="spndefinition.id", default=None, ondelete="CASCADE"
    )
    spn: Optional["SpnDefinition"] = Relationship(back_populates="digital_values")


class AnalogAttributesBase(SQLModel):
    """Represents an Analog Attributes Base."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    is_signed: bool
    is_big_endian: bool
    scale: float
    offset: float
    unit: str


class AnalogAttributes(AnalogAttributesBase, table=True):
    """Represents Analog Attributes."""

    spn: Optional["SpnDefinition"] = Relationship(back_populates="analog_attributes")

