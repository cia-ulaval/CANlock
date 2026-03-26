#!/usr/bin/env python3
"""Visualize attacked CAN dataset.

Usage examples:
  python -m scripts.attack_visualize --pkl data/cache/attacked_window.pkl --out data/cache/attacked_preview.png
  python -m scripts.attack_visualize --pkl data/cache/attacked_window.pkl --spn 184 --out data/cache/spn_184.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from sqlmodel import select

from canlock.db.database import get_session, init_db
from canlock.db.models import PgnDefinition, SpnDefinition, CanMessage
from canlock.decoder import SessionDecoder


def load_attacked(pkl: Path) -> pd.DataFrame:
    if pkl.suffix.lower() in (".txt", ".csv"):
        df = pd.read_csv(pkl)
        
        # 1. Reconversion des timestamps en objets Datetime (LA CORRECTION CLÉ)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        # 2. Parsing du payload
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
            
        # 3. Parsing du CAN ID
        def _parse_id(x):
            if pd.isna(x) or str(x).strip() == "":
                return None
            return int(str(x), 16) if str(x).startswith("0x") else int(x)
            
        if "can_identifier" in df.columns:
            df["can_identifier"] = df["can_identifier"].apply(_parse_id)
    else:
        df = pd.read_pickle(pkl)
        # Sécurité pour le format Pickle aussi
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
    df = df.copy()
    df["pgn"] = df["can_identifier"].apply(lambda x: SessionDecoder.extract_pgn_number_from_payload(int(x)) if not pd.isna(x) else None)
    return df


def plot_payload_preview(df: pd.DataFrame, out: Path, n: int = 500, show: bool = False) -> None:
    df = df.sort_values("timestamp").head(n)
    
    # represent first payload byte as integer
    def first_byte(b):
        # CORRECTION : On assigne -10 aux trames Bus-Off pour qu'elles apparaissent sur le graphique
        if b is None or (isinstance(b, float) and pd.isna(b)):
            return -10  
        bts = bytes(b)
        return int.from_bytes(bts[:1].ljust(1, b"\x00"), "little")

    df["first_byte"] = df["payload"].apply(first_byte)

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Affichage des points avec un peu de transparence (alpha) pour mieux voir les superpositions
    for atk, g in df.groupby(df["attack_type"].fillna("normal")):
        ax.scatter(g["timestamp"], g["first_byte"], label=str(atk), s=15, alpha=0.8)
        
    ax.set_xlabel("timestamp")
    ax.set_ylabel("first payload byte (int)")
    
    # Ajout d'une ligne pointillée pour séparer les trames valides (0-255) des trames supprimées (-10)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # On place la légende à l'extérieur ou en haut à droite pour ne pas cacher les points
    ax.legend(title="attack_type", loc="upper right", bbox_to_anchor=(1.15, 1))
    
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if show:
        plt.show()
    plt.close(fig)


def plot_spn_timeseries(attacked_pkl: Path, spn_id: int, out: Path, show: bool = False) -> None:
    init_db()
    attacked = load_attacked(attacked_pkl)
    if attacked.empty:
        print("No attacked data found")
        return

    min_ts = attacked["timestamp"].min()
    max_ts = attacked["timestamp"].max()

    with get_session() as s:
        # find SPN definition by spn_identifier or by id
        # match by numeric SPN identifier (spn_id passed as int)
        spndef = s.exec(select(SpnDefinition).where(SpnDefinition.spn_identifier == spn_id)).first()
        if not spndef:
            print(f"SPN definition not found for: {spn_id}")
            return

        # load original messages in same window
        q = s.exec(select(CanMessage).where(CanMessage.timestamp >= min_ts, CanMessage.timestamp <= max_ts).order_by(CanMessage.timestamp)).all()
        orig_df = pd.DataFrame([
            {"timestamp": r.timestamp, "can_identifier": r.can_identifier, "payload": r.payload}
            for r in q
        ])

    if orig_df.empty:
        print("No original DB messages found in the same time window")
        return

    rows = []
    # index originals by (timestamp, can_identifier) for quick match
    orig_index = {(r.timestamp, r.can_identifier): r.payload for r in q}

    for idx, row in attacked.iterrows():
        ts = row["timestamp"]
        key = (ts, row.get("can_identifier"))
        orig_payload = orig_index.get(key)
        try:
            attacked_val = SessionDecoder.extract_spn_bits_from_payload(spndef, row.get("payload"))
        except Exception:
            attacked_val = None
        try:
            orig_val = SessionDecoder.extract_spn_bits_from_payload(spndef, orig_payload)
        except Exception:
            orig_val = None
        rows.append({"timestamp": ts, "orig": orig_val, "attacked": attacked_val, "attack_type": row.get("attack_type")})

    df = pd.DataFrame(rows).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["orig"], label="original", alpha=0.6, linewidth=1)
    ax.plot(df["timestamp"], df["attacked"], label="attacked", alpha=0.8, linewidth=1)
    # scatter attacked points with colors by attack_type
    for atk, g in df.groupby(df["attack_type"].fillna("none")):
        ax.scatter(g["timestamp"], g["attacked"], label=str(atk), s=10)

    ax.set_xlabel("timestamp")
    ax.set_ylabel(f"raw SPN value (SPN id {spn_id})")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default="data/cache/attacked_window.pkl")
    p.add_argument("--out", default="data/cache/attacked_preview.png")
    p.add_argument("--spn", type=int, default=None, help="SPN id (spn_identifier or internal id) to plot time series")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    p = Path(args.pkl)
    if not p.exists():
        print(f"Pickle not found: {p}")
        sys.exit(1)
    if args.spn is None:
        df = load_attacked(p)
        print(f"Loaded attacked dataset with {len(df):,} rows")
        plot_payload_preview(df, Path(args.out), n=args.n, show=args.show)
        print(f"Saved preview -> {args.out}")
    else:
        plot_spn_timeseries(p, args.spn, Path(args.out), show=args.show)
        print(f"Saved SPN time series -> {args.out}")


if __name__ == "__main__":
    main()
