from __future__ import annotations

from typing import Optional

from canlock.db.models import SpnDefinition


def _ensure_bytes(payload: Optional[bytes]) -> bytes:
    """Ensure a payload is converted to exactly an 8-byte array.
    
    Args:
        payload: The byte-like payload, which may be None.
        
    Returns:
        The normalized 8-byte payload.
    """
    if payload is None:
        return b"\x00" * 8
    b = bytes(payload)
    if len(b) < 8:
        return b.ljust(8, b"\x00")
    return b


def _get_binary_payload(payload: Optional[bytes]) -> str:
    """Get the string binary representation of a payload.
    
    Args:
        payload: The byte-like payload, which may be None.
        
    Returns:
        A binary string representation of the payload.
    """
    b = _ensure_bytes(payload)
    intval = int(b.hex(), 16) if len(b) > 0 else 0
    return bin(intval)[2:].zfill(len(b) * 8)

def set_spn_bits(payload: Optional[bytes], spn: SpnDefinition, new_raw: int) -> bytes:
    """Return a new dynamic-length payload with the SPN bits replaced by new_raw.
    
    Args:
        payload: The byte-like payload, which may be None.
        spn: The definition mapping of the desired SPN.
        new_raw: The integer value to inject.
        
    Returns:
        The replaced byte representation of the full payload.
    """
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
    """Extract and return the SPN bits as an integer from the given payload.
    
    Args:
        payload: The byte-like payload.
        spn: The definition mapping of the targeted SPN.
        
    Returns:
        The isolated bits for that SPN formatted as an int.
    """
    bin_payload = _get_binary_payload(payload)
    start = spn.bit_start
    length = spn.bit_length
    return int(bin_payload[start : start + length], 2)