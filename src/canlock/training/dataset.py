"""
Dataset PyTorch pour CAN IDS

Gère le fenêtrage, l'extraction des caractéristiques et la normalisation des séquences de messages CAN.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
from typing import Tuple, Optional
import pickle


class CANIDSDataset(Dataset):
    """
    Dataset PyTorch pour la détection d'intrusions sur bus CAN.
    
    Crée des séquences de messages CAN (fenêtres) pour la modélisation temporelle.
    Chaque échantillon est une fenêtre de messages consécutifs avec une seule étiquette.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 50,
        stride: int = 1,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        feature_cols: list = None
    ):
        """
        Initialise le dataset CAN IDS.
        
        Args:
            df: DataFrame avec les colonnes [timestamp, can_identifier, length, payload_bytes, label, label_numeric]
            window_size: Nombre de messages consécutifs dans une fenêtre
            stride: Taille du pas pour la fenêtre glissante
            scaler: Normalisateur pré-ajusté pour la normalisation (None = créer nouveau)
            fit_scaler: Indique s'il faut ajuster le normalisateur sur ces données
            feature_cols: Liste des noms de colonnes de caractéristiques à utiliser
        """
        self.window_size = window_size
        self.stride = stride
        
        # Extraction des caractéristiques
        self.features, self.labels = self._prepare_features(df, feature_cols)
        
        # Normalisation
        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self._fit_scaler()
        else:
            self.scaler = scaler
        
        if fit_scaler or scaler is not None:
            self.features_normalized = self.scaler.transform(
                self.features.reshape(-1, self.features.shape[-1])
            ).reshape(self.features.shape)
        else:
            self.features_normalized = self.features
        
        # Création des fenêtres
        self.windows, self.window_labels = self._create_windows()
        
        print(f"  Created {len(self.windows)} windows from {len(self.features)} messages")
    
    def _prepare_features(
        self, 
        df: pd.DataFrame,
        feature_cols: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrait les caractéristiques du DataFrame.
        
        Les caractéristiques incluent:
        - ID CAN (normalisé)
        - DLC (Data Length Code)
        - Octets de charge utile (8 octets)
        - Priorité (extraite de l'ID CAN)
        - Delta de timestamp (temps écoulé depuis le message précédent)
        
        Returns:
            Tuple de (tableau_caractéristiques, tableau_étiquettes)
        """
        features_list = []
        labels = df['label_numeric'].values
        
        # Calcul des deltas de timestamp
        timestamps = pd.to_datetime(df['timestamp'])
        time_deltas = timestamps.diff().dt.total_seconds().fillna(0).values
        
        for pos_idx, (_, row) in enumerate(df.iterrows()):
            can_id = row['can_identifier']
            dlc = row['length']
            payload_bytes = row['payload_bytes']
            
            # Extraction de la priorité depuis l'ID CAN (bits 26-28 dans J1939)
            priority = (can_id >> 26) & 0x07 if can_id is not None else 0
            
            # Compléter ou tronquer la charge utile à 8 octets
            if len(payload_bytes) < 8:
                payload_bytes = payload_bytes + [0] * (8 - len(payload_bytes))
            else:
                payload_bytes = payload_bytes[:8]
            
            # Combiner toutes les caractéristiques
            feature_vector = [
                float(can_id) if can_id is not None else 0,
                float(dlc) if dlc is not None else 0,
                float(priority),
                time_deltas[pos_idx] * 1000,  # Conversion en millisecondes
            ] + [float(b) for b in payload_bytes]
            
            features_list.append(feature_vector)
        
        features = np.array(features_list, dtype=np.float32)
        
        return features, labels
    
    def _fit_scaler(self):
        """Ajuste le normalisateur sur les caractéristiques."""
        # Aplatir toutes les caractéristiques pour l'ajustement
        features_flat = self.features.reshape(-1, self.features.shape[-1])
        self.scaler.fit(features_flat)
    
    def _create_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des fenêtres glissantes à partir de la séquence de caractéristiques.
        
        Returns:
            Tuple de (tableau_fenêtres, tableau_étiquettes_fenêtres)
        """
        windows = []
        window_labels = []
        
        num_messages = len(self.features_normalized)
        
        for i in range(0, num_messages - self.window_size + 1, self.stride):
            window = self.features_normalized[i:i + self.window_size]
            
            # L'étiquette est l'étiquette la plus commune dans la fenêtre
            # (ou la dernière étiquette pour simplifier)
            label = self.labels[i + self.window_size - 1]
            
            windows.append(window)
            window_labels.append(label)
        
        return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)
    
    def __len__(self) -> int:
        """Retourne le nombre de fenêtres dans le dataset."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtient une fenêtre unique et son étiquette.
        
        Args:
            idx: Index de la fenêtre
            
        Returns:
            Tuple de (tenseur_fenêtre, tenseur_étiquette)
        """
        window = torch.from_numpy(self.windows[idx])
        label = torch.tensor(self.window_labels[idx], dtype=torch.long)
        
        return window, label
    
    def save_scaler(self, path: Path):
        """Sauvegarde le normalisateur ajusté sur le disque."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  # Créer le répertoire parent
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler to {path}")
    
    @staticmethod
    def load_scaler(path: Path):
        """Charge un normalisateur ajusté depuis le disque."""
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_size: int = 50,
    batch_size: int = 128,
    num_workers: int = 0,
    scaler_path: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Crée les dataloaders train/val/test à partir de DataFrames.
    
    Args:
        train_df: DataFrame d'entraînement
        val_df: DataFrame de validation
        test_df: DataFrame de test
        window_size: Taille de fenêtre pour les séquences
        batch_size: Taille de batch pour les dataloaders
        num_workers: Nombre de workers pour le chargement des données
        scaler_path: Chemin pour sauvegarder le normalisateur ajusté
        
    Returns:
        Tuple de (train_loader, val_loader, test_loader, scaler)
    """
    print("Creating datasets...")
    
    # Création du dataset d'entraînement et ajustement du normalisateur
    print("  Train dataset:")
    train_dataset = CANIDSDataset(
        train_df,
        window_size=window_size,
        stride=1,
        fit_scaler=True
    )
    
    # Utilisation du même normalisateur pour validation et test
    print("  Validation dataset:")
    val_dataset = CANIDSDataset(
        val_df,
        window_size=window_size,
        stride=window_size,  # Non-chevauchant pour la validation
        scaler=train_dataset.scaler,
        fit_scaler=False
    )
    
    print("  Test dataset:")
    test_dataset = CANIDSDataset(
        test_df,
        window_size=window_size,
        stride=window_size,  # Non-chevauchant pour le test
        scaler=train_dataset.scaler,
        fit_scaler=False
    )
    
    # Sauvegarde du normalisateur si un chemin est fourni
    if scaler_path:
        train_dataset.save_scaler(scaler_path)
    
    # Création des dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n✓ Dataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, train_dataset.scaler
