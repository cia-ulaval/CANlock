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

import pickle
import random
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
from sqlmodel import select
from torch.utils.data import Dataset

from canlock.attacks.masquerade_attack import MasqueradeAttack
from canlock.attacks.replay_attack import ReplayAttack
from canlock.attacks.spoofing_attack import SpoofingAttack
from canlock.attacks.suspension_attack import SuspensionAttack
from canlock.db.database import get_session, init_db
from canlock.db.models import CanMessage
from canlock.decoder import SessionDecoder


def load_messages_window(
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
    order_by_time: bool = True,
) -> pd.DataFrame:
    """Load CAN messages from the DB into a DataFrame (read-only).

    Args:
        start_ts: Optional ISO-like string parseable by pandas.to_datetime.
        end_ts: Optional ISO-like string parseable by pandas.to_datetime.
        session_id: UUID string to filter by session.
        limit: Maximum number of rows to load.
        order_by_time: Whether to order the results by timestamp.
        
    Returns:
        A pandas DataFrame containing message details.
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

    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "can_identifier": r.can_identifier,
                "length": r.length,
                "payload": r.payload,
            }
            for r in rows
        ]
    )
    return df


def payload_to_tensor(payload, length: int = 8) -> torch.Tensor:
    """Convert payload (bytes or None) to a torch.uint8 tensor of size `length`.
    
    Args:
        payload: The byte payload to convert.
        length: The target length of the tensor, padded with zeros if necessary.
        
    Returns:
        A 1D tensor of uint8 representing the payload.
    """
    import numpy as _np

    if payload is None:
        arr = _np.zeros(length, dtype=_np.uint8)
    else:
        b = bytes(payload)
        arr = _np.frombuffer(b[:length].ljust(length, b"\x00"), dtype=_np.uint8)
    return torch.from_numpy(arr).to(torch.uint8)


def _get_pgn(can_id: Optional[int]) -> Optional[int]:
    """Calculate the Parameter Group Number (PGN) from a CAN identifier.
    
    Args:
        can_id: The integer CAN identifier.
        
    Returns:
        The extracted PGN, or None on failure or if can_id is None.
    """
    try:
        return (
            SessionDecoder.extract_pgn_number_from_payload(int(can_id))
            if can_id is not None
            else None
        )
    except Exception:
        return None


@click.command(help="Generate synthetic CAN attacks and save to pickle (read-only DB)")
@click.option("--start", help="start timestamp (ISO) to filter", default=None)
@click.option("--end", help="end timestamp (ISO) to filter", default=None)
@click.option("--session-id", help="session UUID to filter", default=None)
@click.option("--limit", type=int, default=2000)
@click.option("--out", default="data/cache/attacked_window.pkl")
@click.option("--spoof-rate", type=float, default=0.01)
@click.option(
    "--spoof-sigma-factor",
    type=float,
    default=0.05,
    help="relative sigma factor for spoofing analog perturbation",
)
@click.option(
    "--spoof-min-sigma",
    type=float,
    default=1.0,
    help="minimum sigma for spoofing analog perturbation",
)
@click.option("--suspend-frac", type=float, default=0.05)
@click.option(
    "--tec-increment",
    type=int,
    default=8,
    help="TEC increment per failure for suspension simulation",
)
@click.option(
    "--busoff-threshold",
    type=int,
    default=256,
    help="TEC threshold to consider bus-off",
)
@click.option("--masq-prob", type=float, default=0.2)
@click.option(
    "--attacker-src", default="0x99", help="attacker source address (int/hex)"
)
@click.option("--mode", type=click.Choice(["append", "replace"]), default="append")
@click.option(
    "--replay-rate",
    type=float,
    default=0.0,
    help="fraction of messages/sequences to replay",
)
@click.option(
    "--replay-delay", type=float, default=1.0, help="seconds to delay replayed frames"
)
@click.option(
    "--replay-seq",
    is_flag=True,
    help="replay contiguous sequences instead of isolated frames",
)
@click.option(
    "--replay-length",
    type=int,
    default=1,
    help="sequence length when --replay-seq is set",
)
@click.option(
    "--spn",
    type=int,
    default=None,
    help="SPN identifier to target (single-SPN attacks)",
)
@click.option(
    "--attack-order",
    type=str,
    default="spoofing,masquerade,replay,suspension",
    help="Comma-separated order in which to apply attacks. Valid names: spoofing, masquerade, replay, suspension",
)
def main(
    start,
    end,
    session_id,
    limit,
    out,
    spoof_rate,
    spoof_sigma_factor,
    spoof_min_sigma,
    suspend_frac,
    tec_increment,
    busoff_threshold,
    masq_prob,
    attacker_src,
    mode,
    replay_rate,
    replay_delay,
    replay_seq,
    replay_length,
    spn,
    attack_order,
) -> None:
    """Main CLI entrypoint for attack synthesis.
    
    Loads a dataset from the database, applies a sequence of CAN attacks,
    and saves the attacked dataset to a file.
    
    Args:
        start: Start timestamp to filter the messages.
        end: End timestamp to filter the messages.
        session_id: Optional session UUID.
        limit: Limit message loading.
        out: Output file path.
        spoof_rate: Injected messages rate for spoofing.
        spoof_sigma_factor: Factor for sigma perturbation.
        spoof_min_sigma: Min sigma perturbation.
        suspend_frac: Fraction of messages suspended.
        tec_increment: Simulated TEC increment.
        busoff_threshold: Threshold for turning bus-off.
        masq_prob: Masquerade probability.
        attacker_src: Source address of attacker.
        mode: Injection mode.
        replay_rate: Rate of replay.
        replay_delay: Replay delay in seconds.
        replay_seq: Whether to replay sequences.
        replay_length: Length of sequences to replay.
        spn: Target SPN for specific targeting.
        attack_order: Order of attacks.
    """
    init_db()

    df = load_messages_window(
        start_ts=start,
        end_ts=end,
        session_id=session_id,
        limit=limit,
    )
    print(f"Loaded {len(df):,} messages from DB (read-only)")

    attacker_src_int = (
        int(attacker_src, 0) if isinstance(attacker_src, str) else attacker_src
    )

    # Build attack objects (one class per attack)
    spoof = SpoofingAttack(
        injection_rate=spoof_rate,
        mode=mode,
        sigma_factor=spoof_sigma_factor,
        min_sigma=spoof_min_sigma,
    )
    susp = SuspensionAttack(
        suspend_fraction=suspend_frac,
        tec_increment=tec_increment,
        bus_off_threshold=busoff_threshold,
    )
    masq = MasqueradeAttack(attacker_source=attacker_src_int, prob=masq_prob)
    replay = ReplayAttack(
        replay_rate=replay_rate,
        delay_seconds=replay_delay,
        replay_sequence=replay_seq,
        sequence_length=replay_length,
    )

    # Application séquentielle selon l'ordre demandé
    attack_funcs = {
        "spoofing": lambda d: spoof.apply(d, target=spn),
        "masquerade": lambda d: masq.apply(d, target=spn),
        "replay": lambda d: replay.apply(d, target=spn),
        "suspension": lambda d: susp.apply(d, target=spn),
    }

    # normalize and validate order tokens
    order_tokens = [t.strip().lower() for t in attack_order.split(",") if t.strip()]
    # accept short synonyms
    syn_map = {"spoof": "spoofing", "masq": "masquerade", "susp": "suspension"}
    normalized = []
    for t in order_tokens:
        t2 = syn_map.get(t, t)
        if t2 not in attack_funcs:
            raise SystemExit(
                f"Invalid attack name in --attack-order: {t} (valid: {', '.join(attack_funcs.keys())})"
            )
        normalized.append(t2)

    # apply attacks in requested order
    df_current = df
    for atk in normalized:
        df_current = attack_funcs[atk](df_current)

    df_final = df_current

    # 1. Mise à jour de la colonne 'length'
    def _actual_length(p):
        if p is None or pd.isna(p):
            return 0  # Trame supprimée (Bus-Off)
        return len(p)

    df_final["length"] = df_final["payload"].apply(_actual_length)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Support writing either a pickle (.pkl) or a human-readable text/CSV (.txt/.csv)
    if out_path.suffix.lower() in (".txt", ".csv"):
        out_df = df_final.copy()

        # 2. Recalculer 'pgn' et 'src' à partir du 'can_identifier' final
        def _extract_pgn(cid):
            if pd.isna(cid):
                return None
            try:
                return SessionDecoder.extract_pgn_number_from_payload(int(cid))
            except Exception:
                return None

        out_df["pgn"] = out_df["can_identifier"].apply(_extract_pgn)
        out_df["src"] = out_df["can_identifier"].apply(
            lambda x: int(x) & 0xFF if not pd.isna(x) else None
        )

        # 3. Convertir les charges utiles avec gestion explicite des trames supprimées (Bus-Off)
        def _hex_payload(b):
            if b is None or pd.isna(b):
                return "BUS_OFF"
            return b.hex()

        out_df["payload_hex"] = out_df.get(
            "payload", pd.Series([None] * len(out_df))
        ).apply(_hex_payload)

        # 4. Formater les identifiants en Hexadécimal pour la lisibilité
        out_df["can_identifier"] = out_df["can_identifier"].apply(
            lambda x: f"0x{int(x):08X}" if not pd.isna(x) else ""
        )
        out_df["pgn"] = out_df["pgn"].apply(
            lambda x: f"0x{int(x):04X}" if not pd.isna(x) else ""
        )
        out_df["src"] = out_df["src"].apply(
            lambda x: f"0x{int(x):02X}" if not pd.isna(x) else ""
        )

        # 5. Nettoyer et réorganiser les colonnes
        out_df.drop(columns=["payload"], inplace=True, errors="ignore")

        cols_order = [
            "timestamp",
            "can_identifier",
            "pgn",
            "src",
            "length",
            "attack_type",
            "payload_hex",
            "replay_source_index",
            "replay_delay_s",
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
    print(
        df_final.get("attack_type", pd.Series(dtype=object)).value_counts(dropna=True)
    )


if __name__ == "__main__":
    main()
