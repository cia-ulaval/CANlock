import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class CANDataModule(pl.LightningDataModule):
    """DataModule Lightning pour les données CAN."""

    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        super().__init__()
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_train = torch.LongTensor(y_train)
        self.y_test = torch.LongTensor(y_test)
        self.batch_size = batch_size

        # Validation set séparé (si fourni, sinon utilise X_test)
        if X_val is not None and y_val is not None:
            self.X_val = torch.FloatTensor(X_val)
            self.y_val = torch.LongTensor(y_val)
        else:
            self.X_val = self.X_test
            self.y_val = self.y_test

    def train_dataloader(self) -> DataLoader:
        # Entraînement uniquement sur les données normales
        normal_mask = self.y_train == 0
        train_dataset = TensorDataset(
            self.X_train[normal_mask], self.y_train[normal_mask]
        )
        return DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self) -> DataLoader:
        # Validation sur le set de validation dédié
        val_dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        test_dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)
