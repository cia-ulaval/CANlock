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
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.preprocessing import StandardScaler
from sqlmodel import select

# Ignorer certains warnings
warnings.filterwarnings("ignore")

from canlock.attacks.attack_ddos import DDoSAttack
from canlock.attacks.masquerade_attack import MasqueradeAttack
from canlock.attacks.replay_attack import ReplayAttack
from canlock.attacks.spoofing_attack import SpoofingAttack
from canlock.attacks.suspension_attack import SuspensionAttack
from canlock.data.can_data_module import CANDataModule
from canlock.db.database import get_session, init_db
from canlock.db.models import CanMessage
from canlock.models.anomaly_detector import AnomalyDetector
from canlock.models.cnn_lstm_autoencoder import CnnLstmAutoencoder
from canlock.models.rnn_vae import RnnVae


def payload_to_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Convertit les messages CAN bruts en features numériques.
    Chaque message → [can_id, byte_0..byte_7, delta_t, freq] = 11 features.
    - can_id : identifiant CAN
    - byte_0..byte_7 : octets du payload
    - delta_t : temps écoulé depuis le dernier message (secondes)
    - freq : nombre de messages avec le même CAN ID dans une fenêtre glissante
    Retourne (features, labels) où label=1 si attack_type != "normal".
    """
    # Pré-calculer les delta timestamps
    timestamps = df["timestamp"].values
    can_ids = df["can_identifier"].values

    # Delta timestamp : temps depuis le message précédent
    delta_ts = np.zeros(len(df))
    delta_ts[1:] = np.diff(timestamps.astype(np.float64) / 1e9)  # en secondes
    delta_ts[0] = 0.0
    # Clipper les outliers au 99e percentile
    delta_ts = np.clip(delta_ts, 0, np.percentile(delta_ts, 99))

    # Fréquence par CAN ID : compteur glissant sur les 50 derniers messages
    FREQ_WINDOW = 50
    freq = np.zeros(len(df))
    for i in range(len(df)):
        start = max(0, i - FREQ_WINDOW)
        freq[i] = np.sum(can_ids[start:i + 1] == can_ids[i])
    freq = np.clip(freq, 0, np.percentile(freq, 99))

    rows = []
    labels = []
    for idx, (_, msg) in enumerate(df.iterrows()):
        if msg.get("payload") is None:
            continue
        try:
            payload = (
                bytes(msg["payload"])
                if not isinstance(msg["payload"], bytes)
                else msg["payload"]
            )
        except (TypeError, ValueError):
            continue
        payload_bytes = list(payload[:8]) + [0] * max(0, 8 - len(payload))
        row = [int(msg["can_identifier"])] + payload_bytes + [delta_ts[idx], freq[idx]]
        rows.append(row)
        labels.append(0 if msg.get("attack_type", "normal") == "normal" else 1)
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)


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
# MLflow options
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default="http://localhost:5050",
    help="URI du serveur MLflow (défaut: http://localhost:5050)",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="Nom de l'expérience MLflow (défaut: nom du modèle)",
)
@click.option(
    "--target-recall",
    type=float,
    default=None,
    help="Recall cible pour le seuil de détection (ex: 0.99). Si omis, optimise le F1.",
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
    mlflow_tracking_uri: str,
    mlflow_experiment: str | None,
    target_recall: float | None,
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

    # 2. Split temporel des messages bruts (80% train+val / 20% test)
    split_idx = int(len(df_raw_messages) * 0.8)
    df_train_val_raw = df_raw_messages.iloc[:split_idx].copy()
    df_test_raw = df_raw_messages.iloc[split_idx:].copy()

    df_train_val_raw["attack_type"] = "normal"
    df_test_raw["attack_type"] = "normal"

    click.echo(
        f"[+] Split temporel : {len(df_train_val_raw)} train+val / {len(df_test_raw)} test"
    )

    # 3. Génération des attaques sur la portion test uniquement
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

    # 4. Extraction des features depuis les payloads bruts
    click.echo("[*] Extraction des features depuis les payloads bruts (CAN ID + 8 octets)...")
    normal_features, normal_labels_raw = payload_to_features(df_train_val_raw)
    attacked_features, attacked_labels = payload_to_features(df_attacked)
    click.echo(f"[+] Features normales : {normal_features.shape} ({normal_features.shape[1]} features)")
    click.echo(f"[+] Features attaquées : {attacked_features.shape}")

    # 5. Normalisation
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_features)
    attacked_scaled = scaler.transform(attacked_features)

    # 6. Création de séquences
    normal_sequences = create_sequences(normal_scaled, seq_len=seq_len, stride=stride)
    normal_labels = np.zeros(len(normal_sequences), dtype=np.int64)

    # Séquences attaquées (test)
    attacked_sequences, attacked_seq_labels = create_sequences_with_labels(
        attacked_scaled, attacked_labels, seq_len=seq_len, stride=stride
    )

    # 7. Split séquentiel : 75% train / 15% val / 10% test-normal
    n_train = int(len(normal_sequences) * 0.75)
    n_val = int(len(normal_sequences) * 0.15)
    X_train = normal_sequences[:n_train]
    y_train = normal_labels[:n_train]
    X_val = normal_sequences[n_train : n_train + n_val]
    y_val = normal_labels[n_train : n_train + n_val]

    # Test = normales restantes (10%) + toutes les attaques
    # → normales pour mesurer faux positifs, attaques pour vrais positifs
    X_test_normal = normal_sequences[n_train + n_val :]
    y_test_normal = normal_labels[n_train + n_val :]
    X_test = np.concatenate([X_test_normal, attacked_sequences], axis=0)
    y_test = np.concatenate([y_test_normal, attacked_seq_labels], axis=0)

    click.echo(f"[+] Séquences d'entraînement (normales) : {X_train.shape}")
    click.echo(f"[+] Séquences de validation (normales) : {X_val.shape}")
    click.echo(
        f"[+] Séquences de test : {X_test.shape} "
        f"({(y_test == 0).sum()} normales, {(y_test == 1).sum()} anomalies)"
    )

    # 8. DataModule et Modèle
    datamodule = CANDataModule(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        batch_size=batch_size,
        X_val=X_val,
        y_val=y_val,
    )

    n_features = X_train.shape[2]

    if model_type == "cnn_lstm":
        model = CnnLstmAutoencoder(n_features=n_features, seq_len=seq_len, lr=lr)
        monitor_metric = "val_loss"
    else:
        model = RnnVae(n_features=n_features, seq_len=seq_len, lr=lr, kl_weight=1e-3)
        monitor_metric = "val_loss"

    # 9. MLflow Logger
    experiment_name = mlflow_experiment or model_type
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
    )
    click.echo(f"[*] MLflow tracking : {mlflow_tracking_uri} (expérience: {experiment_name})")

    # 10. Entraînement
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{model_type}-{{epoch:02d}}-{{{monitor_metric}:.2f}}",
        save_top_k=1,
        monitor=monitor_metric,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric, min_delta=0.00, patience=20, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        logger=mlflow_logger,
        gradient_clip_val=1.0,
    )

    click.echo(f"[*] Démarrage de l'entraînement du modèle '{model_type}'...")
    trainer.fit(model, datamodule=datamodule)

    click.echo("[*] Démarrage de la phase de test avec attaques (sur le dataset de test)...")
    trainer.test(model, datamodule=datamodule)

    # 11. Évaluation : détection d'anomalies
    click.echo("[*] Évaluation de la détection d'anomalies (seuil optimal, métriques)...")
    detector = AnomalyDetector(model=model, datamodule=datamodule, device=str(device))
    errors, targets = detector.compute_reconstruction_errors()

    # Seuil F1-optimal
    threshold_f1, best_f1 = detector.find_optimal_threshold(errors, targets)
    results_f1 = detector.evaluate(errors, targets, threshold_f1)

    click.echo(f"[+] === Seuil F1-optimal : {threshold_f1:.6f} (F1={best_f1:.4f}) ===")
    click.echo(f"[+] ROC-AUC : {results_f1['auc']:.4f}")
    click.echo(f"[+] Matrice de confusion : TP={results_f1['tp']} FP={results_f1['fp']} TN={results_f1['tn']} FN={results_f1['fn']}")
    click.echo(f"[+] FPR={results_f1['fpr']:.4f}  Recall={results_f1['recall']:.4f}")
    click.echo(results_f1["classification_report"])

    # Seuil recall-cible (si demandé)
    if target_recall is not None:
        threshold_recall, actual_recall = detector.find_threshold_for_recall(
            errors, targets, target_recall
        )
        results_recall = detector.evaluate(errors, targets, threshold_recall)

        click.echo(f"[+] === Seuil recall>={target_recall} : {threshold_recall:.6f} ===")
        click.echo(f"[+] Matrice de confusion : TP={results_recall['tp']} FP={results_recall['fp']} TN={results_recall['tn']} FN={results_recall['fn']}")
        click.echo(f"[+] FPR={results_recall['fpr']:.4f}  Recall={results_recall['recall']:.4f}")
        click.echo(results_recall["classification_report"])

    click.echo(
        f"[+] Entraînement et test terminés. Le meilleur modèle est à : {checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    train()
