"""Inspect and validate attacked CAN dataset saved by attack_synthesis.

Provides a readable preview and validation that attacks target a single SPN.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Optional

import pandas as pd

from canlock.db.database import get_session, init_db
from canlock.db.models import PgnDefinition, SpnDefinition, CanMessage
from canlock.decoder import SessionDecoder
from sqlmodel import select


def format_payload_hex(b: Optional[bytes]) -> str:
    if b is None:
        return "<None>"
    return "0x" + b.hex()


def preview(attacked_pkl: Path, n: int = 10):
    # Support pickle or CSV/text exports (CSV contains `payload_hex` column)
    if attacked_pkl.suffix.lower() in (".txt", ".csv"):
        df = pd.read_csv(attacked_pkl)
        
        # 1. Parsing du payload
        def _to_bytes(s):
            if pd.isna(s) or str(s).strip() in ("", "BUS_OFF"):
                return None
            s2 = str(s).strip()
            if s2.startswith("0x"):
                s2 = s2[2:]
            try:
                return bytes.fromhex(s2)
            except Exception:
                return None
                
        if "payload_hex" in df.columns:
            df["payload"] = df["payload_hex"].apply(_to_bytes)
            
        # 2. Parsing du CAN ID de l'Hexadécimal vers l'Entier
        def _parse_id(x):
            if pd.isna(x) or str(x).strip() == "":
                return None
            return int(str(x), 16) if str(x).startswith("0x") else int(x)
            
        if "can_identifier" in df.columns:
            df["can_identifier"] = df["can_identifier"].apply(_parse_id)
    else:
        df = pd.read_pickle(attacked_pkl)
    # Add derived columns
    df = df.copy()
    df["pgn"] = df["can_identifier"].apply(lambda x: SessionDecoder.extract_pgn_number_from_payload(int(x)) if x is not None else None)
    df["can_id_hex"] = df["can_identifier"].apply(lambda x: f"0x{x:08X}" if x is not None else None)
    df["payload_hex"] = df["payload"].apply(format_payload_hex)
    df["src"] = df["can_identifier"].apply(lambda x: int(x) & 0xFF if x is not None else None)

    print("Preview (first %d rows):" % n)
    display_cols = ["timestamp", "can_id_hex", "pgn", "src", "length", "payload_hex", "attack_type"]
    print(df[display_cols].head(n).to_string(index=False))
    print("\nAttack counts:\n", df["attack_type"].value_counts(dropna=False))
    return df


def validate_single_spn(attacked_pkl: Path) -> None:
    """Validate that attacks affect a single SPN.

    Strategy:
    - Load attacked DataFrame
    - Load original messages from DB for the same time window
    - For attacked rows that have a matching original (same timestamp and can_identifier),
      compare extracted SPN raw values per SPN definition for that PGN.
    - Collect SPN ids that changed and report.
    """
    init_db()
    # Load attacked dataset (pickle or CSV/text)
    if attacked_pkl.suffix.lower() in (".txt", ".csv"):
        attacked = pd.read_csv(attacked_pkl)
        
        # 1. Parsing du payload
        def _to_bytes(s):
            if pd.isna(s) or str(s).strip() in ("", "BUS_OFF"):
                return None
            s2 = str(s).strip()
            if s2.startswith("0x"):
                s2 = s2[2:]
            try:
                return bytes.fromhex(s2)
            except Exception:
                return None
                
        if "payload_hex" in attacked.columns:
            attacked["payload"] = attacked["payload_hex"].apply(_to_bytes)
            
        # 2. Parsing du CAN ID de l'Hexadécimal vers l'Entier
        def _parse_id(x):
            if pd.isna(x) or str(x).strip() == "":
                return None
            return int(str(x), 16) if str(x).startswith("0x") else int(x)
            
        if "can_identifier" in attacked.columns:
            attacked["can_identifier"] = attacked["can_identifier"].apply(_parse_id)
    else:
        attacked = pd.read_pickle(attacked_pkl)
    if attacked.empty:
        print("No attacked data found in pickle")
        return

    min_ts = attacked["timestamp"].min()
    max_ts = attacked["timestamp"].max()

    # Load original messages from DB in same window (may be large; limit to 100k)
    with get_session() as s:
        q = s.exec(
            select(CanMessage).where(CanMessage.timestamp >= min_ts, CanMessage.timestamp <= max_ts).order_by(CanMessage.timestamp)
        ).all()
        orig_df = pd.DataFrame([
            {"timestamp": r.timestamp, "can_identifier": r.can_identifier, "length": r.length, "payload": r.payload}
            for r in q
        ])

    if orig_df.empty:
        print("No original DB messages found in the same time window")
        return

    changed_spns = set()

    # Cache PGN->spns
    pgn_spn_cache: dict[int, list[SpnDefinition]] = {}

    for idx, row in attacked.iterrows():
        if pd.isna(row.get("attack_type")):
            continue
        ts = row["timestamp"]
        # find candidate original rows with same timestamp
        candidates = orig_df[orig_df["timestamp"] == ts]
        if candidates.empty:
            continue
        # pick first matching original
        orig = candidates.iloc[0]

        # compute PGN from original can_identifier
        if orig["can_identifier"] is None:
            continue
        pgn = SessionDecoder.extract_pgn_number_from_payload(int(orig["can_identifier"]))

        if pgn not in pgn_spn_cache:
            with get_session() as s:
                pgn_def = s.exec(select(PgnDefinition).where(PgnDefinition.pgn_identifier == pgn)).first()
                if pgn_def:
                    spns = s.exec(select(SpnDefinition).where(SpnDefinition.pgn_id == pgn_def.id)).all()
                else:
                    spns = []
            pgn_spn_cache[pgn] = spns

        spns = pgn_spn_cache[pgn]
        if not spns:
            continue

        # For each SPN in this PGN, extract raw bits and compare
        for spn in spns:
            try:
                orig_val = SessionDecoder.extract_spn_bits_from_payload(spn, orig["payload"])
            except Exception:
                continue
            try:
                attacked_val = SessionDecoder.extract_spn_bits_from_payload(spn, row["payload"])
            except Exception:
                continue
            if orig_val != attacked_val:
                changed_spns.add(spn.spn_identifier if spn.spn_identifier is not None else spn.id)

    if not changed_spns:
        print("No SPN value changes detected between DB and attacked dataset for matched rows.")
        return

    print("SPNs changed by attacks:")
    for s in sorted(changed_spns):
        print(" - ", s)

    if len(changed_spns) == 1:
        print("Validation PASSED: only one SPN was affected.")
    else:
        print(f"Validation FAILED: {len(changed_spns)} different SPNs were affected.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default="data/cache/attacked_window.pkl")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--n", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    p = Path(args.pkl)
    if not p.exists():
        print(f"Pickle not found: {p}")
        return
    if args.preview:
        preview(p, args.n)
    if args.validate:
        validate_single_spn(p)


if __name__ == "__main__":
    main()
