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
from canlock.db.models import DefinedDigitalValues


class SpoofingAttack(AttackBase):
    """Spoofing attack that injects or replaces payloads for a single signal/PGN.

    Note: `target` is interpreted by this class as a PGN identifier if provided.
    In future iterations this can be extended to target a SPN specifically.
    """

    def __init__(self, injection_rate: float = 0.01, mode: str = "append", sigma_factor: float = 0.05, min_sigma: float = 1.0, seed: Optional[int] = None):
        super().__init__("spoofing")
        self.injection_rate = float(injection_rate)
        self.mode = mode
        self.sigma_factor = float(sigma_factor)
        self.min_sigma = float(min_sigma)
        if seed is not None:
            random.seed(seed)

    def _get_pgn(self, can_id: Optional[int]) -> Optional[int]:
        try:
            return SessionDecoder.extract_pgn_number_from_payload(int(can_id)) if can_id is not None else None
        except Exception:
            return None

    def apply(self, df: pd.DataFrame, target: Optional[int] = None) -> pd.DataFrame:
        if self.injection_rate <= 0:
            return df
        
        df2 = df.copy()
        pgns = [self._get_pgn(x) for x in df2["can_identifier"].tolist()]
        df2["pgn"] = pgns

        # If target is provided we interpret it as an SPN identifier and
        # only modify the bits corresponding to that SPN (single-signal attack).
        spn_def = None
        if target is not None:
            with get_session() as s:
                spn_row = s.exec(select(SpnDefinition).where(SpnDefinition.spn_identifier == target)).first()
                if spn_row:
                    # build a small detached object carrying necessary attributes
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

        n_inject = max(1, int(len(candidates) * self.injection_rate)) if candidates else 0
        chosen = random.sample(candidates, n_inject) if n_inject and len(candidates) >= n_inject else candidates

        spoofed_rows = []
        for idx in chosen:
            row = df2.loc[idx].copy()
            if spn_def is not None:
                # create a plausible value based on SPN definition
                try:
                    orig_raw = get_spn_bits(row.get("payload"), spn_def)
                except Exception:
                    orig_raw = None

                # analog SPN: jitter the physical value slightly
                if spn_def.is_analog and getattr(spn_def, "analog_attributes", None):
                    an = spn_def.analog_attributes
                    if orig_raw is not None:
                        orig_phys = an.scale * orig_raw + an.offset
                        # small gaussian perturbation: configurable factor or min_sigma
                        sigma = max(abs(orig_phys) * self.sigma_factor, self.min_sigma)
                        new_phys = orig_phys + random.gauss(0, sigma)
                    else:
                        # choose a mid-range guess if no original: use offset
                        new_phys = an.offset
                    new_raw = int(round((new_phys - an.offset) / an.scale)) if an.scale != 0 else 0
                else:
                    # digital or unknown: pick a different defined digital value if available
                    new_raw = None
                    with get_session() as s:
                        choices = s.exec(select(DefinedDigitalValues).where(DefinedDigitalValues.spn_id == spn_def.id)).all()
                    if choices:
                        # pick a random defined value (prefer different from original)
                        orig_raw = orig_raw if orig_raw is not None else -1
                        vals = [c.value for c in choices]
                        cand = [v for v in vals if v != orig_raw]
                        if cand:
                            new_raw = int(random.choice(cand))
                        else:
                            new_raw = int(random.choice(vals))
                    else:
                        # fallback: random raw within bit range
                        new_raw = random.randint(0, (1 << spn_def.bit_length) - 1)

                # clip to bit length
                mask_max = (1 << spn_def.bit_length) - 1
                new_raw = max(0, min(mask_max, int(new_raw)))
                row["payload"] = set_spn_bits(row.get("payload"), spn_def, new_raw)
            else:
                # fallback: randomize some bytes in the payload
                orig = row["payload"] if row["payload"] is not None else b"\x00" * 8
                orig_bytes = bytes(orig)[:8].ljust(8, b"\x00")
                arr = bytearray(orig_bytes)
                for _ in range(random.randint(1, 3)):
                    i = random.randrange(0, 8)
                    arr[i] = arr[i] ^ random.getrandbits(8)
                row["payload"] = bytes(arr)
            row["attack_type"] = self.name
            spoofed_rows.append(row)
            if self.mode == "replace":
                df2.loc[idx, "payload"] = row["payload"]
                df2.loc[idx, "attack_type"] = self.name

        if self.mode == "append" and spoofed_rows:
            df2 = pd.concat([df2, pd.DataFrame(spoofed_rows)], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        return df2
