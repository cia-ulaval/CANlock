#!/usr/bin/env python3
"""Generate synthetic CAN attacks

This script reads a time/window of CAN messages from the project's SQL database
and produces a synthetic attacked dataset saved as a pickle file. It never
modifies the database (read-only).

Usage (example):
  python -m scripts.attack_synthesis --limit 2000 --out data/cache/attacked_window.pkl
  uv run python -m scripts.attack_synthesis --limit 5000 --out data/cache/test_attacked.csv --spn 161 --spoof-rate 0.5 --suspend-frac 0 --masq-prob 0; uv run python -m scripts.attack_inspect --pkl data/cache/test_attacked.csv --preview --validate --n 15; uv run python -m scripts.attack_visualize --pkl data/cache/test_attacked.csv --out data/cache/test_preview.png
"""
from __future__ import annotations

import argparse
import random
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sqlmodel import select

from canlock.db.database import get_session, init_db
from canlock.db.models import CanMessage
from canlock.decoder import SessionDecoder

from canlock.attacks.SpoofingAttack import SpoofingAttack
from canlock.attacks.SuspensionAttack import SuspensionAttack
from canlock.attacks.MasqueradeAttack import MasqueradeAttack
from canlock.attacks.ReplayAttack import ReplayAttack


def load_messages_window(start_ts: Optional[str] = None,
                         end_ts: Optional[str] = None,
                         session_id: Optional[str] = None,
                         limit: Optional[int] = None,
                         order_by_time: bool = True) -> pd.DataFrame:
    """Load CAN messages from the DB into a DataFrame (read-only).

    start_ts / end_ts: ISO-like strings parseable by pandas.to_datetime
    session_id: UUID string to filter by session
    limit: maximum number of rows to load
    """
    with get_session() as s:
        q = select(CanMessage)
        if session_id is not None:
            q = q.where(CanMessage.session_id == session_id)
        if start_ts is not None:
            q = q.where(CanMessage.timestamp >= pd.to_datetime(start_ts))
        if end_ts is not None:
            q = q.where(CanMessage.timestamp <= pd.to_datetime(end_ts))
        if order_by_time:
            q = q.order_by(CanMessage.timestamp)
        if limit is not None:
            q = q.limit(limit)
        rows = s.exec(q).all()

    df = pd.DataFrame([
        {
            "timestamp": r.timestamp,
            "can_identifier": r.can_identifier,
            "length": r.length,
            "payload": r.payload,
        }
        for r in rows
    ])
    return df


def payload_to_tensor(payload, length: int = 8) -> torch.Tensor:
    """Convert payload (bytes or None) to a torch.uint8 tensor of size `length`."""
    import numpy as _np

    if payload is None:
        arr = _np.zeros(length, dtype=_np.uint8)
    else:
        b = bytes(payload)
        arr = _np.frombuffer(b[:length].ljust(length, b"\x00"), dtype=_np.uint8)
    return torch.from_numpy(arr).to(torch.uint8)


def _get_pgn(can_id: Optional[int]) -> Optional[int]:
    try:
        return SessionDecoder.extract_pgn_number_from_payload(int(can_id)) if can_id is not None else None
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic CAN attacks and save to pickle (read-only DB)")
    p.add_argument("--start", help="start timestamp (ISO) to filter", default=None)
    p.add_argument("--end", help="end timestamp (ISO) to filter", default=None)
    p.add_argument("--session-id", help="session UUID to filter", default=None)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--out", default="data/cache/attacked_window.pkl")
    p.add_argument("--spoof-rate", type=float, default=0.01)
    p.add_argument("--suspend-frac", type=float, default=0.05)
    p.add_argument("--masq-prob", type=float, default=0.2)
    p.add_argument("--attacker-src", type=lambda x: int(x, 0), default=0x99, help="attacker source address (int/hex)")
    p.add_argument("--mode", choices=["append", "replace"], default="append")
    p.add_argument("--spn", type=int, default=None, help="SPN identifier to target (single-SPN attacks)")
    return p.parse_args()


def main() -> None:
    init_db()
    args = parse_args()

    df = load_messages_window(start_ts=args.start, end_ts=args.end, session_id=args.session_id, limit=args.limit)
    print(f"Loaded {len(df):,} messages from DB (read-only)")

    # Build attack objects (one class per attack)
    spoof = SpoofingAttack(injection_rate=args.spoof_rate)
    susp = SuspensionAttack(suspend_fraction=args.suspend_frac)
    masq = MasqueradeAttack(attacker_source=args.attacker_src, prob=args.masq_prob)
    replay = ReplayAttack(replay_rate=0.0, delay_seconds=1.0) 

    # Application séquentielle propre
    df_spoofed = spoof.apply(df, target=args.spn)
    df_masq = masq.apply(df_spoofed, target=args.spn)
    df_replay = replay.apply(df_masq, target=args.spn)
    df_final = susp.apply(df_replay, target=args.spn)

    # 1. Mise à jour de la colonne 'length'
    def _actual_length(p):
        if p is None or pd.isna(p):
            return 0  # Trame supprimée (Bus-Off)
        return len(p)
    
    df_final["length"] = df_final["payload"].apply(_actual_length)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Support writing either a pickle (.pkl) or a human-readable text/CSV (.txt/.csv)
    if out_path.suffix.lower() in (".txt", ".csv"):
        out_df = df_final.copy()

        # 2. Recalculer 'pgn' et 'src' à partir du 'can_identifier' final
        def _extract_pgn(cid):
            if pd.isna(cid): return None
            try:
                return SessionDecoder.extract_pgn_number_from_payload(int(cid))
            except Exception:
                return None

        out_df["pgn"] = out_df["can_identifier"].apply(_extract_pgn)
        out_df["src"] = out_df["can_identifier"].apply(lambda x: int(x) & 0xFF if not pd.isna(x) else None)

        # 3. Convertir les charges utiles avec gestion explicite des trames supprimées (Bus-Off)
        def _hex_payload(b):
            if b is None or pd.isna(b):
                return "BUS_OFF"
            return b.hex()
        
        out_df["payload_hex"] = out_df.get("payload", pd.Series([None]*len(out_df))).apply(_hex_payload)

        # 4. Formater les identifiants en Hexadécimal pour la lisibilité
        out_df["can_identifier"] = out_df["can_identifier"].apply(lambda x: f"0x{int(x):08X}" if not pd.isna(x) else "")
        out_df["pgn"] = out_df["pgn"].apply(lambda x: f"0x{int(x):04X}" if not pd.isna(x) else "")
        out_df["src"] = out_df["src"].apply(lambda x: f"0x{int(x):02X}" if not pd.isna(x) else "")

        # 5. Nettoyer et réorganiser les colonnes
        out_df.drop(columns=["payload"], inplace=True, errors="ignore")
        
        cols_order = [
            "timestamp", "can_identifier", "pgn", "src", "length", 
            "attack_type", "payload_hex", "replay_source_index", "replay_delay_s"
        ]
        final_cols = [c for c in cols_order if c in out_df.columns]
        out_df = out_df[final_cols]

        out_df.to_csv(out_path, index=False)
        print(f"Saved attacked dataset (CSV) -> {out_path}")
    else:
        with open(out_path, "wb") as f:
            pickle.dump(df_final, f) 
        print(f"Saved attacked dataset -> {out_path}")
        
    print("Attack counts:")
    print(df_final.get("attack_type", pd.Series(dtype=object)).value_counts(dropna=True))


if __name__ == "__main__":
    main()