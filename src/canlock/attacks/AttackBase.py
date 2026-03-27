from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from canlock.db.models import SpnDefinition


def _ensure_bytes(payload: Optional[bytes]) -> bytes:
    if payload is None:
        return b"\x00" * 8
    b = bytes(payload)
    if len(b) < 8:
        return b.ljust(8, b"\x00")
    return b


def _get_binary_payload(payload: Optional[bytes]) -> str:
    b = _ensure_bytes(payload)
    intval = int(b.hex(), 16) if len(b) > 0 else 0
    return bin(intval)[2:].zfill(len(b) * 8)

def set_spn_bits(payload: Optional[bytes], spn: SpnDefinition, new_raw: int) -> bytes:
    """Return a new dynamic-length payload with the SPN bits replaced by new_raw."""
    b = _ensure_bytes(payload)
    bin_payload = _get_binary_payload(b)
    
    start = spn.bit_start
    length = spn.bit_length
    mask_max = (1 << length) - 1
    
    if new_raw is None:
        new_raw = 0
    if new_raw < 0 or new_raw > mask_max:
        raise ValueError("new_raw does not fit in spn bit_length")
        
    new_bits = bin(new_raw)[2:].zfill(length)
    
    # Remplacement exact au niveau des bits
    new_bin = bin_payload[:start] + new_bits + bin_payload[start + length :]
    new_int = int(new_bin, 2)
    
    # Reconversion en octets en respectant la longueur nécessaire
    byte_length = (len(new_bin) + 7) // 8
    return new_int.to_bytes(byte_length, "big")


def get_spn_bits(payload: Optional[bytes], spn: SpnDefinition) -> int:
    bin_payload = _get_binary_payload(payload)
    start = spn.bit_start
    length = spn.bit_length
    return int(bin_payload[start : start + length], 2)


class AttackBase(ABC):
    """Abstract base class for attack algorithms.

    Subclasses must implement `apply(df, target)` where `target` is the identifier
    of the signal to attack (SPN, PGN or a custom filter). For now the
    implementation expects the caller to provide a DataFrame and a `target`
    that the subclass understands.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, df: pd.DataFrame, target: Optional[int] = None) -> pd.DataFrame:
        """Apply attack to the DataFrame and return a new DataFrame.

        Args:
            df: input DataFrame with CAN messages
            target: identifier of the signal to attack (semantics left to subclass)
        Returns:
            df_attacked: DataFrame with modifications (attack annotations)
        """
        raise NotImplementedError()
