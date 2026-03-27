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

    def __init__(self, suspend_fraction: float = 0.1, tec_increment: int = 8, bus_off_threshold: int = 256, recovery_delay_s: float = 0.05, seed: Optional[int] = None):
        super().__init__("suspension")
        self.suspend_fraction = float(suspend_fraction)
        self.tec_increment = int(tec_increment)
        self.bus_off_threshold = int(bus_off_threshold)
        self.recovery_delay_s = float(recovery_delay_s) # Délai de 50ms par défaut
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
            # compute src per message (LSB of CAN id)
            def _src(can_id):
                if can_id is None:
                    return None
                return int(can_id) & 0xFF

            df2["src"] = df2["can_identifier"].apply(_src)
            pgn_candidates = [i for i in candidates]
            
            # select the victim source as the most common src among candidates
            srcs = [df2.loc[i, "src"] for i in pgn_candidates if df2.loc[i, "src"] is not None]
            if not srcs:
                return df2
            counts = Counter(srcs)
            victim_src = counts.most_common(1)[0][0]

            # 1. On identifie les messages spécifiquement ciblés par l'attaque
            victim_target_idxs = set(i for i in pgn_candidates if df2.loc[i, "src"] == victim_src)
            
            # 2. On récupère TOUS les messages de la victime pour simuler le temps qui passe
            victim_all_idxs = df2.index[df2["src"] == victim_src].tolist()
            victim_all_idxs.sort(key=lambda x: pd.to_datetime(df2.loc[x, "timestamp"]))

            TEC = 0
            last_bus_off_time = None

            for idx in victim_all_idxs:
                current_time = pd.to_datetime(df2.loc[idx, "timestamp"])

                # ÉVALUATION DE LA RECONNEXION
                if TEC >= self.bus_off_threshold and last_bus_off_time is not None:
                    # Si le délai (ex: 50ms) est écoulé, l'ECU redémarre
                    if (current_time - last_bus_off_time).total_seconds() >= self.recovery_delay_s:
                        TEC = 0
                        last_bus_off_time = None

                # APPLICATION DES EFFETS
                if TEC < self.bus_off_threshold:
                    # L'ECU est en ligne. L'attaquant perturbe-t-il CE message ?
                    if idx in victim_target_idxs:
                        df2.loc[idx, "payload"] = None
                        df2.loc[idx, "attack_type"] = "suspension"
                        TEC += self.tec_increment
                        
                        # Si l'injection d'erreur le fait passer en Bus-Off, on note l'heure
                        if TEC >= self.bus_off_threshold:
                            last_bus_off_time = current_time
                else:
                    # L'ECU est en Bus-Off. Il n'arrive pas à envoyer ce message.
                    df2.loc[idx, "payload"] = None
                    df2.loc[idx, "attack_type"] = "bus_off"

        else:
            # fallback behavior: remove entire payload if no SPN target resolution
            df2.loc[df2.index.isin(to_modify), "payload"] = None
            df2.loc[df2.index.isin(to_modify), "attack_type"] = self.name
            
        return df2