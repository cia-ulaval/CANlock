from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from .AttackBase import AttackBase, set_spn_bits, get_spn_bits
from canlock.decoder import SessionDecoder
from canlock.db.database import get_session
from canlock.db.models import PgnDefinition, SpnDefinition
from sqlmodel import select
import types
from collections import Counter


class SuspensionAttack(AttackBase):
    """Suspension attack that masks (removes) a fraction of messages for a target PGN/SPN."""

    def __init__(self, suspend_fraction: float = 0.1, seed: Optional[int] = None):
        super().__init__("suspension")
        self.suspend_fraction = float(suspend_fraction)
        if seed is not None:
            random.seed(seed)

    def _get_pgn(self, can_id: Optional[int]) -> Optional[int]:
        try:
            return SessionDecoder.extract_pgn_number_from_payload(int(can_id)) if can_id is not None else None
        except Exception:
            return None

    def apply(self, df: pd.DataFrame, target: Optional[int] = None) -> pd.DataFrame:
        if self.suspend_fraction <= 0:
            return df
        
        df2 = df.copy()
        pgns = [self._get_pgn(x) for x in df2["can_identifier"].tolist()]
        df2["pgn"] = pgns

        # Interpret target as SPN identifier: zero-out only the SPN bits for chosen messages
        spn_def = None
        if target is not None:
            with get_session() as s:
                spn_row = s.exec(select(SpnDefinition).where(SpnDefinition.spn_identifier == target)).first()
                if spn_row:
                    # detached simple object
                    an = None
                    if spn_row.analog_attributes:
                        an = types.SimpleNamespace(scale=spn_row.analog_attributes.scale, offset=spn_row.analog_attributes.offset)
                    spn_def = types.SimpleNamespace(id=spn_row.id, bit_length=spn_row.bit_length, bit_start=spn_row.bit_start, is_analog=spn_row.is_analog, analog_attributes=an, pgn_id=spn_row.pgn_id)
                    pgn_def = s.exec(select(PgnDefinition).where(PgnDefinition.id == spn_row.pgn_id)).first()
                    target_pgn = pgn_def.pgn_identifier if pgn_def else None
                else:
                    target_pgn = None
            if target_pgn is not None:
                candidates = df2.index[df2["pgn"] == target_pgn].tolist()
            else:
                candidates = []
        else:
            candidates = df2.index.tolist()

        n_remove = int(len(candidates) * self.suspend_fraction)
        to_modify = set(random.sample(candidates, n_remove)) if n_remove else set()
        if spn_def is not None:
            # Simulate Bus-Off style suspension per-description:
            # - choose the victim source (ECU) as the most frequent source for this PGN
            # - for messages from that source, simulate repeated error injections: each failed attempt
            #   increments a TEC-like counter by 8; when >=256 mark bus-off and remove further messages
            # compute src per message (LSB of CAN id)
            def _src(can_id):
                if can_id is None:
                    return None
                return int(can_id) & 0xFF

            df2["src"] = df2["can_identifier"].apply(_src)
            # candidate indices for the target PGN
            pgn_candidates = [i for i in candidates]
            # select the victim source as the most common src among candidates
            srcs = [df2.loc[i, "src"] for i in pgn_candidates if df2.loc[i, "src"] is not None]
            if not srcs:
                return df2
            counts = Counter(srcs)
            victim_src = counts.most_common(1)[0][0]

            # get ordered indices for messages from victim_src and target PGN
            victim_idxs = [i for i in pgn_candidates if df2.loc[i, "src"] == victim_src]
            # sort by timestamp to simulate chronological attempts
            victim_idxs.sort(key=lambda x: pd.to_datetime(df2.loc[x, "timestamp"]))

            TEC = 0
            BUS_OFF_THRESHOLD = 256
            for idx in victim_idxs:
                if TEC < BUS_OFF_THRESHOLD:
                    # simulate injected bit error causing the frame to fail -> payload effectively lost
                    df2.loc[idx, "payload"] = None
                    df2.loc[idx, "attack_type"] = "suspension"  # error injection event
                    TEC += 8
                else:
                    # ECU is now bus-off; subsequent frames are not sent -> mark them as bus_off
                    df2.loc[idx, "payload"] = None
                    df2.loc[idx, "attack_type"] = "bus_off"
            # For completeness, mark any later messages from same victim as bus_off as well
            remaining_idxs = [i for i in df2.index if df2.loc[i, "src"] == victim_src and i not in victim_idxs]
            for i in remaining_idxs:
                df2.loc[i, "payload"] = None
                df2.loc[i, "attack_type"] = "bus_off"
        else:
            # fallback behavior: remove entire payload if no SPN target resolution
            df2.loc[df2.index.isin(to_modify), "payload"] = None
            df2.loc[df2.index.isin(to_modify), "attack_type"] = self.name
        return df2
