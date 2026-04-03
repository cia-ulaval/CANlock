import torch
import torch.nn as nn
import pytorch_lightning as pl

class CnnLstmAutoencoder(pl.LightningModule):
    """
    Autoencoder pour la détection d'anomalies sur données CAN.
    - Encoder: CNN pour extraction de features + LSTM pour dépendances temporelles
    - Decoder: LSTM + CNN transposé pour reconstruction
    """
    def __init__(self, n_features: int, seq_len: int, lr: float = 1e-3):
        super().__init__()
        if seq_len < 4:
            raise ValueError(f"seq_len must be >= 4, got {seq_len}")
        self.save_hyperparameters()
        self.n_features = n_features
        self.seq_len = seq_len
        self.lr = lr

        # Encoder CNN
        self.enc_conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm1d(32)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)

        # Encoder LSTM
        self.enc_lstm = nn.LSTM(64, 32, num_layers=1, batch_first=True, bidirectional=True)

        # Bottleneck
        self.bottleneck = nn.Linear(64, 32)

        # Decoder LSTM
        self.dec_lstm = nn.LSTM(32, 64, num_layers=1, batch_first=True, bidirectional=True)

        # Decoder CNN (transposed)
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec_bn1 = nn.BatchNorm1d(64)
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec_bn2 = nn.BatchNorm1d(32)
        self.dec_conv3 = nn.Conv1d(32, n_features, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.pool(x)
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len//4, 64)
        x, _ = self.enc_lstm(x)
        x = self.dropout(x)
        # Bottleneck: prendre le dernier état
        x = self.bottleneck(x[:, -1, :])  # (batch, 32)
        return x

    def decode(self, z):
        # z: (batch, 32)
        # Expand pour LSTM
        reduced_len = self.seq_len // 4
        z = z.unsqueeze(1).repeat(1, reduced_len, 1)  # (batch, seq_len//4, 32)
        z, _ = self.dec_lstm(z)  # (batch, seq_len//4, 128)
        z = z.permute(0, 2, 1)  # (batch, 128, seq_len//4)
        z = self.relu(self.dec_bn1(self.dec_conv1(z)))
        z = self.relu(self.dec_bn2(self.dec_conv2(z)))
        z = self.dec_conv3(z)
        # Interpolate to match exact input sequence length
        z = nn.functional.interpolate(z, size=self.seq_len, mode='linear', align_corners=False)
        z = z.permute(0, 2, 1)  # (batch, seq_len, n_features)
        return z

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def _compute_loss(self, batch):
        x, _ = batch  # On ignore les labels pour l'entraînement non-supervisé
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }
