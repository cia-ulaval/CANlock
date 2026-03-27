from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from .AttackBase import AttackBase, get_spn_bits, set_spn_bits
from canlock.decoder import SessionDecoder
from canlock.db.database import get_session
from canlock.db.models import SpnDefinition, PgnDefinition
from sqlmodel import select
import types


class MasqueradeAttack(AttackBase):
    """Masquerade attack that changes the source address (LSB 8 bits) for selected messages."""

    def __init__(self, attacker_source: Optional[int] = None, prob: float = 0.2, seed: Optional[int] = None):
        super().__init__("masquerade")
        self.attacker_source = attacker_source
        self.prob = float(prob)
        if seed is not None:
            random.seed(seed)

    def _src(self, can_id: Optional[int]) -> Optional[int]:
        if can_id is None:
            return None
        return int(can_id) & 0xFF

    def apply(self, df: pd.DataFrame, target: Optional[int] = None) -> pd.DataFrame:
        if self.prob <= 0:
            return df
        
        df2 = df.copy()
        srcs = [self._src(x) for x in df2["can_identifier"].tolist()]
        df2["src"] = srcs

        # If a target SPN is given, limit masquerade to messages carrying that SPN's PGN
        target_pgn = None
        if target is not None:
            with get_session() as s:
                spn_row = s.exec(select(SpnDefinition).where(SpnDefinition.spn_identifier == target)).first()
                if spn_row:
                    an = None
                    if spn_row.analog_attributes:
                        an = types.SimpleNamespace(scale=spn_row.analog_attributes.scale, offset=spn_row.analog_attributes.offset)
                    # create detached spn object for later use
                    spn_def = types.SimpleNamespace(id=spn_row.id, bit_length=spn_row.bit_length, bit_start=spn_row.bit_start, is_analog=spn_row.is_analog, analog_attributes=an, pgn_id=spn_row.pgn_id)
                    pgn_def = s.exec(select(PgnDefinition).where(PgnDefinition.id == spn_row.pgn_id)).first()
                    target_pgn = pgn_def.pgn_identifier if pgn_def else None

        if target is None and target_pgn is None:
            candidates = df2.index.tolist()
        elif target_pgn is not None:
            pgns = [SessionDecoder.extract_pgn_number_from_payload(int(x)) if x is not None else None for x in df2["can_identifier"].tolist()]
            df2["pgn"] = pgns
            candidates = df2.index[df2["pgn"] == target_pgn].tolist()
        else:
            candidates = df2.index[df2["src"] == target].tolist()

        chosen = [i for i in candidates if random.random() < self.prob]
        for i in chosen:
            cid = int(df2.loc[i, "can_identifier"]) if df2.loc[i, "can_identifier"] is not None else 0
            new_cid = (cid & ~0xFF) | (self.attacker_source & 0xFF) if self.attacker_source is not None else cid
            # ESCALADE DE PRIORITÉ : Forcer les 3 bits de priorité (26-28) à 0 (Priorité maximale)
            # Le masque ~(0x7 << 26) efface les bits de priorité
            new_cid = new_cid & ~(0x7 << 26)
            df2.loc[i, "can_identifier"] = new_cid
            # If targeting a specific SPN, also adjust SPN bits slightly to look plausible
            if target is not None:
                try:
                    with get_session() as s:
                        spn_def = s.exec(select(SpnDefinition).where(SpnDefinition.spn_identifier == target)).first()
                    if spn_def:
                        try:
                            orig_raw = get_spn_bits(df2.loc[i, "payload"], spn_def)
                        except Exception:
                            orig_raw = None
                        if spn_def.is_analog and getattr(spn_def, "analog_attributes", None):
                            an = spn_def.analog_attributes
                            if orig_raw is not None:
                                orig_phys = an.scale * orig_raw + an.offset
                                sigma = max(abs(orig_phys) * 0.03, 0.5)
                                new_phys = orig_phys + random.gauss(0, sigma)
                                new_raw = int(round((new_phys - an.offset) / an.scale)) if an.scale != 0 else 0
                            else:
                                new_raw = orig_raw if orig_raw is not None else 0
                        else:
                            # digital: leave payload or pick a close value
                            new_raw = orig_raw if orig_raw is not None else 0
                        # clip and set
                        mask_max = (1 << spn_def.bit_length) - 1
                        new_raw = max(0, min(mask_max, int(new_raw)))
                        df2.loc[i, "payload"] = set_spn_bits(df2.loc[i, "payload"], spn_def, new_raw)
                except Exception:
                    pass
            df2.loc[i, "attack_type"] = self.name
        return df2
