import sys
from pathlib import Path

# Add src to sys.path if not running from root
sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd().parent / "src"))

import warnings

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sqlmodel import select
from tqdm import tqdm

# Ignorer certains warnings
warnings.filterwarnings("ignore")

from canlock.attacks.attack_ddos import DDoSAttack
from canlock.attacks.masquerade_attack import MasqueradeAttack
from canlock.attacks.replay_attack import ReplayAttack
from canlock.attacks.spoofing_attack import SpoofingAttack
from canlock.attacks.suspension_attack import SuspensionAttack
from canlock.data.can_data_module import CANDataModule
from canlock.db.database import get_session, init_db
from canlock.db.models import CanMessage, PgnDefinition, SpnDefinition
from canlock.decoder import SessionDecoder
from canlock.models.cnn_lstm_autoencoder import CnnLstmAutoencoder
from canlock.models.rnn_vae import RnnVae


def decode_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Décode un DataFrame de messages CAN bruts en valeurs SPN.
    Préserve la colonne attack_type si présente.
    """
    with get_session() as session:
        decoder = SessionDecoder(db=session)

        # Construire le cache PGN/SPN
        pgn_cache = {}
        decoded_rows = []

        for _, msg in tqdm(df.iterrows(), total=len(df), desc="Décodage"):
            if msg.get("can_identifier") is None or msg.get("payload") is None:
                continue
            if pd.isna(msg["can_identifier"]) or pd.isna(msg.get("payload", None)):
                continue

            try:
                payload = (
                    bytes(msg["payload"])
                    if not isinstance(msg["payload"], bytes)
                    else msg["payload"]
                )
            except (TypeError, ValueError):
                continue

            pgn_number = SessionDecoder.extract_pgn_number_from_payload(
                int(msg["can_identifier"])
            )

            if pgn_number not in pgn_cache:
                pgn_def = session.exec(
                    select(PgnDefinition).where(
                        PgnDefinition.pgn_identifier == pgn_number
                    )
                ).first()

                if pgn_def:
                    spns = session.exec(
                        select(SpnDefinition).where(SpnDefinition.pgn_id == pgn_def.id)
                    ).all()
                    pgn_cache[pgn_number] = [
                        (spn, spn.analog_attributes) for spn in spns
                    ]
                else:
                    pgn_cache[pgn_number] = None

            spn_rules = pgn_cache[pgn_number]
            if spn_rules:
                spns = [s[0] for s in spn_rules]
                analog_attrs = [s[1] for s in spn_rules]
                values = decoder.extract_values_from_spns(spns, analog_attrs, payload)
                if values:
                    row = {"timestamp": msg["timestamp"]}
                    row.update(values)
                    if "attack_type" in msg:
                        row["attack_type"] = msg["attack_type"]
                    decoded_rows.append(row)

    return pd.DataFrame(decoded_rows)


def create_sequences_with_labels(data, labels, seq_len, stride):
    """Crée des séquences à partir de données en série temporelle avec étiquettes"""
    sequences = []
    seq_labels = []
    for i in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[i : i + seq_len])
        # L'étiquette de séquence est 1 s'il y a n'importe quelle attaque dans la fenêtre
        seq_labels.append(1 if np.any(labels[i : i + seq_len]) else 0)
    return np.array(sequences), np.array(seq_labels)


def create_sequences(data, seq_len, stride):
    """Crée des séquences à partir de données non étiquetées"""
    sequences = []
    for i in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[i : i + seq_len])
    return np.array(sequences)


@click.command()
@click.option(
    "--model-type",
    type=click.Choice(["cnn_lstm", "rnn_vae"]),
    required=True,
    help="Le modèle à entraîner (cnn_lstm ou rnn_vae)",
)
@click.option(
    "--limit",
    type=int,
    default=50000,
    help="Nombre de messages bruts à charger depuis la BD (défaut: 50000)",
)
@click.option(
    "--batch-size", type=int, default=32, help="Taille des batchs (défaut: 32)"
)
@click.option(
    "--epochs",
    type=int,
    default=50,
    help="Nombre maximum d'époques pour l'entraînement (défaut: 50)",
)
@click.option(
    "--lr",
    type=float,
    default=1e-3,
    help="Taux d'apprentissage au départ (défaut: 0.001)",
)
@click.option(
    "--seq-len",
    type=int,
    default=30,
    help="Taille des séquences temporelles (défaut: 30)",
)
@click.option(
    "--stride",
    type=int,
    default=30,
    help="Décalage (stride) des fenêtres de séquence (défaut: 30)",
)
@click.option(
    "--save-dir",
    type=str,
    default="checkpoints",
    help="Répertoire de sauvegarde des checkpoints",
)
# SpoofingAttack options
@click.option(
    "--spoof-injection-rate",
    type=float,
    default=0.03,
    help="Taux d'injection pour SpoofingAttack",
)
@click.option(
    "--spoof-mode",
    type=click.Choice(["append", "replace"]),
    default="replace",
    help="Mode pour SpoofingAttack",
)
@click.option(
    "--spoof-sigma-factor",
    type=float,
    default=0.05,
    help="Facteur sigma pour SpoofingAttack",
)
@click.option(
    "--spoof-min-sigma",
    type=float,
    default=1.0,
    help="Sigma minimum pour SpoofingAttack",
)
# MasqueradeAttack options
@click.option(
    "--masq-attacker-source",
    type=int,
    default=153,
    help="Source attaquant pour MasqueradeAttack (défaut: 153/0x99)",
)
@click.option(
    "--masq-prob", type=float, default=0.02, help="Probabilité pour MasqueradeAttack"
)
# ReplayAttack options
@click.option(
    "--replay-rate", type=float, default=0.03, help="Taux de rejeu pour ReplayAttack"
)
@click.option(
    "--replay-delay-seconds", type=float, default=1.0, help="Délai pour ReplayAttack"
)
@click.option(
    "--replay-sequence",
    is_flag=True,
    default=False,
    help="Rejouer en séquence pour ReplayAttack",
)
@click.option(
    "--replay-sequence-length",
    type=int,
    default=1,
    help="Longueur de séquence pour ReplayAttack",
)
@click.option(
    "--replay-preserve-interval",
    is_flag=True,
    default=True,
    help="Préserver l'intervalle pour ReplayAttack",
)
# SuspensionAttack options
@click.option(
    "--susp-suspend-fraction",
    type=float,
    default=0.02,
    help="Fraction de suspension pour SuspensionAttack",
)
@click.option(
    "--susp-tec-increment",
    type=int,
    default=8,
    help="Incrément TEC pour SuspensionAttack",
)
@click.option(
    "--susp-bus-off-threshold",
    type=int,
    default=256,
    help="Seuil Bus-Off pour SuspensionAttack",
)
@click.option(
    "--susp-recovery-delay-s",
    type=float,
    default=0.05,
    help="Délai de récupération pour SuspensionAttack",
)
# DDoSAttack options
@click.option(
    "--ddos-repetitions",
    type=int,
    default=1000,
    help="Nombre de répétitions pour DDoSAttack",
)
@click.option(
    "--ddos-interval",
    type=float,
    default=0.0001,
    help="Intervalle pour DDoSAttack",
)
def train(
    model_type: str,
    limit: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seq_len: int,
    stride: int,
    save_dir: str,
    spoof_injection_rate: float,
    spoof_mode: str,
    spoof_sigma_factor: float,
    spoof_min_sigma: float,
    masq_attacker_source: int,
    masq_prob: float,
    replay_rate: float,
    replay_delay_seconds: float,
    replay_sequence: bool,
    replay_sequence_length: int,
    replay_preserve_interval: bool,
    susp_suspend_fraction: float,
    susp_tec_increment: int,
    susp_bus_off_threshold: int,
    susp_recovery_delay_s: float,
    ddos_repetitions: int,
    ddos_interval: float,
) -> None:
    """Script de lancement d'entraînement des modèles de détection d'anomalies CANlock."""

    pl.seed_everything(42)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    click.echo(f"[*] Appareil utilisé : {device}")

    # 1. Chargement des données brutes
    click.echo(f"[*] Chargement de {limit} messages bruts depuis la base de données...")
    init_db()
    with get_session() as session:
        rows = session.exec(
            select(CanMessage).order_by(CanMessage.timestamp).limit(limit)
        ).all()

    df_raw_messages = pd.DataFrame(
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
    click.echo(f"[+] Messages bruts chargés : {len(df_raw_messages)}")

    # 2. Partitionnement des données (Train/Valid = Normal SEULEMENT, Test = Attaques + Normal)
    split_idx = int(len(df_raw_messages) * 0.8)
    df_train_valid_raw = df_raw_messages.iloc[:split_idx].copy()
    df_test_raw = df_raw_messages.iloc[split_idx:].copy()

    df_train_valid_raw["attack_type"] = "normal"
    df_test_raw["attack_type"] = "normal"

    click.echo(
        "[*] Génération des attaques synthétiques (spoofing, masquerade, replay, suspension, ddos) sur le set de test..."
    )
    df_attacked = df_test_raw.copy()

    spoof = SpoofingAttack(
        injection_rate=spoof_injection_rate,
        mode=spoof_mode,
        sigma_factor=spoof_sigma_factor,
        min_sigma=spoof_min_sigma,
    )
    masq = MasqueradeAttack(attacker_source=masq_attacker_source, prob=masq_prob)
    replay = ReplayAttack(
        replay_rate=replay_rate,
        delay_seconds=replay_delay_seconds,
        replay_sequence=replay_sequence,
        sequence_length=replay_sequence_length,
        preserve_interval=replay_preserve_interval,
    )
    susp = SuspensionAttack(
        suspend_fraction=susp_suspend_fraction,
        tec_increment=susp_tec_increment,
        bus_off_threshold=susp_bus_off_threshold,
        recovery_delay_s=susp_recovery_delay_s,
    )
    ddos = DDoSAttack(
        signal_name="can_identifier",
        repetitions=ddos_repetitions,
        interval=ddos_interval,
    )

    df_attacked = spoof.apply(df_attacked, target=None)
    df_attacked = masq.apply(df_attacked, target=None)
    df_attacked = replay.apply(df_attacked, target=None)
    df_attacked = susp.apply(df_attacked, target=None)
    
    df_attacked = ddos.apply(df_attacked, target=None)
    df_attacked["attack_type"] = df_attacked["attack_type"].fillna("DDoS")

    df_attacked = df_attacked.sort_values("timestamp").reset_index(drop=True)
    click.echo(f"[+] Messages de test après attaques : {len(df_attacked)}")

    # 3. Décodage en SPN
    click.echo("[*] Décodage des messages d'entraînement normaux en valeurs SPN...")
    df_normal_decoded = decode_raw_dataframe(df_train_valid_raw)

    click.echo("[*] Décodage des messages de test (avec attaques) en valeurs SPN...")
    df_attacked_decoded = decode_raw_dataframe(df_attacked)

    # 4. Sélection des colonnes SPN communes et nettoyage
    meta_cols = {"timestamp", "attack_type"}
    normal_spn_cols = set(df_normal_decoded.columns) - meta_cols
    attacked_spn_cols = set(df_attacked_decoded.columns) - meta_cols
    common_cols = sorted(normal_spn_cols & attacked_spn_cols)

    # Sélection des 15 meilleures colonnes par taux de remplissage
    fill_rates = (
        df_normal_decoded[common_cols].notna().mean().sort_values(ascending=False)
    )
    top_cols = fill_rates.head(15).index.tolist()
    click.echo(f"[+] 15 colonnes SPN retenues : {len(top_cols)}")

    df_normal_clean = df_normal_decoded[top_cols].ffill().bfill().dropna()
    df_attacked_clean = df_attacked_decoded[top_cols + ["attack_type"]].copy()
    df_attacked_clean[top_cols] = df_attacked_clean[top_cols].ffill().bfill()
    df_attacked_clean = df_attacked_clean.dropna(subset=top_cols)

    # 5. Normalisation et création de séquences
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(df_normal_clean.values)
    attacked_scaled = scaler.transform(df_attacked_clean[top_cols].values)

    attacked_labels = (df_attacked_clean["attack_type"] != "normal").astype(int).values

    X_train_valid = create_sequences(normal_scaled, seq_len=seq_len, stride=stride)
    y_train_valid = np.zeros(len(X_train_valid), dtype=np.int64)

    # Split Train / Valid (80/20 of the normal sequences)
    n_train_seqs = int(len(X_train_valid) * 0.8)
    X_train_full = X_train_valid[:n_train_seqs]
    y_train_full = y_train_valid[:n_train_seqs]
    X_val_full = X_train_valid[n_train_seqs:]
    y_val_full = y_train_valid[n_train_seqs:]

    # Validation / Test part
    X_test_full, y_test_full = create_sequences_with_labels(
        attacked_scaled, attacked_labels, seq_len=seq_len, stride=stride
    )

    click.echo(f"[+] Séquences d'entraînement (normales) : {X_train_full.shape}")
    click.echo(f"[+] Séquences de validation (normales) : {X_val_full.shape}")
    click.echo(f"[+] Séquences de test (avec anomalies) : {X_test_full.shape}")

    # 6. DataModule et Modèle
    datamodule = CANDataModule(
        X_train=X_train_full,
        X_test=X_test_full,
        y_train=y_train_full,
        y_test=y_test_full,
        batch_size=batch_size,
        X_val=X_val_full,
        y_val=y_val_full,
    )

    n_features = X_train_full.shape[2]

    if model_type == "cnn_lstm":
        model = CnnLstmAutoencoder(n_features=n_features, seq_len=seq_len, lr=lr)
        monitor_metric = "val_loss"
    else:
        model = RnnVae(n_features=n_features, seq_len=seq_len, lr=lr, kl_weight=1e-3)
        monitor_metric = "val_loss"

    # 7. Entraînement
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{model_type}-{{epoch:02d}}-{{{monitor_metric}:.2f}}",
        save_top_k=1,
        monitor=monitor_metric,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric, min_delta=0.00, patience=10, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )

    click.echo(f"[*] Démarrage de l'entraînement du modèle '{model_type}'...")
    trainer.fit(model, datamodule=datamodule)

    click.echo("[*] Démarrage de la phase de test avec attaques (sur le dataset de test)...")
    trainer.test(model, datamodule=datamodule)

    click.echo(
        f"[+] Entraînement et test terminés. Le meilleur modèle est à : {checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    train()
