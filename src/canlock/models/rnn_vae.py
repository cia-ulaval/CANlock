import pytorch_lightning as pl
import torch
import torch.nn as nn


class RnnVae(pl.LightningModule):
    """
    Variational Auto-Encoder avec encodeur/décodeur LSTM
    pour la détection d'anomalies sur données CAN.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 1,
        lr: float = 1e-3,
        kl_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.lr = lr
        self.kl_weight = kl_weight

        # Encoder 
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_dropout = nn.Dropout(0.2)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder 
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * 2)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder_dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(hidden_dim * 2, n_features)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features)
        output, _ = self.encoder_lstm(x) 
        output = self.encoder_dropout(output)
        # Prendre le dernier timestep
        h = output[:, -1, :]  
        mu = self.fc_mu(h)  
        logvar = self.fc_logvar(h)  
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_to_hidden(z)  
        # Répéter pour chaque timestep
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1) 
        output, _ = self.decoder_lstm(h) 
        output = self.decoder_dropout(output)
        x_hat = self.output_layer(output)  
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def _compute_loss(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _ = batch
        x_hat, mu, logvar = self(x)

        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.kl_weight * kl_loss
        return loss, recon_loss, kl_loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, recon_loss, kl_loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_loss", kl_loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, recon_loss, kl_loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, recon_loss, kl_loss = self._compute_loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_recon_loss", recon_loss)
        self.log("test_kl_loss", kl_loss)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
