from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from .AttackBase import AttackBase
from canlock.decoder import SessionDecoder
from canlock.db.database import get_session
from canlock.db.models import SpnDefinition, PgnDefinition
from sqlmodel import select
import types


class ReplayAttack(AttackBase):
    """Replay attack: capture valid messages and retransmit them later.

    Behavior:
    - If `target` is an SPN identifier, capture payloads for the PGN carrying that SPN.
    - Re-insert (append) copies of captured frames later in the trace with a configurable delay.
    - Uses the same CAN ID (legitimate ID) so messages look syntactically valid.
    """

    def __init__(self, replay_rate: float = 0.05, delay_seconds: float = 1.0, replay_sequence: bool = False, sequence_length: int = 1, preserve_interval: bool = True, seed: Optional[int] = None):
        super().__init__("replay")
        self.replay_rate = float(replay_rate)
        self.delay_seconds = float(delay_seconds)
        self.replay_sequence = bool(replay_sequence)
        self.sequence_length = int(sequence_length)
        self.preserve_interval = bool(preserve_interval)
        if seed is not None:
            random.seed(seed)

    def _get_pgn(self, can_id: Optional[int]) -> Optional[int]:
        try:
            return SessionDecoder.extract_pgn_number_from_payload(int(can_id)) if can_id is not None else None
        except Exception:
            return None

    def apply(self, df: pd.DataFrame, target: Optional[int] = None) -> pd.DataFrame:
        if self.replay_rate <= 0:
            return df
        
        df2 = df.copy()
        # compute PGNs for messages
        pgns = [self._get_pgn(x) for x in df2["can_identifier"].tolist()]
        df2["pgn"] = pgns

        target_pgn = None
        if target is not None:
            with get_session() as s:
                spn_row = s.exec(select(SpnDefinition).where(SpnDefinition.spn_identifier == target)).first()
                if spn_row:
                    pgn_def = s.exec(select(PgnDefinition).where(PgnDefinition.id == spn_row.pgn_id)).first()
                    target_pgn = pgn_def.pgn_identifier if pgn_def else None

        if target_pgn is None and target is not None:
            # no matching SPN/PGN found
            return df2

        if target is None:
            candidates = df2.index.tolist()
        else:
            candidates = df2.index[df2["pgn"] == target_pgn].tolist()

        if not candidates:
            return df2

        n_replay = max(1, int(len(candidates) * self.replay_rate))
        chosen = random.sample(candidates, n_replay) if len(candidates) >= n_replay else candidates

        replayed_rows = []
        if self.replay_sequence and self.sequence_length > 1:
            # replay contiguous sequences of length `sequence_length`
            max_start = len(df2) - self.sequence_length
            seq_candidates = [i for i in candidates if i <= max_start]
            if not seq_candidates:
                return df2
            chosen_starts = random.sample(seq_candidates, min(len(seq_candidates), n_replay))
            for start in chosen_starts:
                seq_idxs = list(range(start, start + self.sequence_length))
                base_ts = pd.to_datetime(df2.loc[start, "timestamp"]) if not pd.isna(df2.loc[start, "timestamp"]) else None
                # compute deltas if preserving intervals
                deltas = []
                if self.preserve_interval and base_ts is not None:
                    for j in seq_idxs:
                        deltas.append(pd.to_datetime(df2.loc[j, "timestamp"]) - base_ts)
                for j, orig_idx in enumerate(seq_idxs):
                    row = df2.loc[orig_idx].copy()
                    try:
                        if base_ts is not None and self.preserve_interval:
                            new_ts = base_ts + deltas[j] + pd.to_timedelta(self.delay_seconds, unit="s")
                            row["timestamp"] = pd.Timestamp(new_ts)
                        else:
                            row["timestamp"] = pd.Timestamp(pd.to_datetime(row["timestamp"]) + pd.to_timedelta(self.delay_seconds, unit="s"))
                    except Exception:
                        row["timestamp"] = row.get("timestamp")
                    row["attack_type"] = self.name
                    row["replay_source_index"] = int(orig_idx)
                    row["replay_delay_s"] = float(self.delay_seconds)
                    replayed_rows.append(row)
        else:
            for idx in chosen:
                row = df2.loc[idx].copy()
                try:
                    ts = pd.to_datetime(row["timestamp"]) + pd.to_timedelta(self.delay_seconds, unit="s")
                    row["timestamp"] = pd.Timestamp(ts)
                except Exception:
                    row["timestamp"] = row.get("timestamp")
                row["attack_type"] = self.name
                row["replay_source_index"] = int(idx)
                row["replay_delay_s"] = float(self.delay_seconds)
                replayed_rows.append(row)

        if replayed_rows:
            df2 = pd.concat([df2, pd.DataFrame(replayed_rows)], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        return df2
