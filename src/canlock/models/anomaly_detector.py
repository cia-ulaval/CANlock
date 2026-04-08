import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


class AnomalyDetector:
    """
    Classe utilitaire pour la détection d'anomalies par reconstruction.
    Gère le calcul des erreurs, le choix du seuil optimal et l'évaluation.
    """

    def __init__(self, model, datamodule, device="cpu"):
        self.model = model
        self.datamodule = datamodule
        self.device = device

    def compute_reconstruction_errors(self) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)
        reconstruction_errors = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in self.datamodule.test_dataloader():
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                x_hat = output[0] if isinstance(output, tuple) else output
                mse = torch.mean((batch_x - x_hat) ** 2, dim=(1, 2))
                reconstruction_errors.extend(mse.cpu().numpy())
                all_targets.extend(batch_y.numpy())
        return np.array(reconstruction_errors), np.array(all_targets)

    def find_optimal_threshold(
        self, errors: np.ndarray, targets: np.ndarray
    ) -> tuple[float, float]:
        if len(np.unique(targets)) < 2:
            raise ValueError(
                "targets must contain at least two classes to compute an optimal threshold"
            )
        precision, recall, thresholds = precision_recall_curve(targets, errors)
        if len(thresholds) == 0:
            raise ValueError("precision_recall_curve returned no thresholds")
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)

        if len(thresholds) == 0:
            raise ValueError("thresholds variable is empty.")

        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        )
        return optimal_threshold, f1_scores[optimal_idx]

    def find_threshold_for_recall(
        self, errors: np.ndarray, targets: np.ndarray, target_recall: float = 0.99
    ) -> tuple[float, float]:
        """Trouve le seuil le plus élevé garantissant au moins target_recall."""
        if len(np.unique(targets)) < 2:
            raise ValueError(
                "targets must contain at least two classes to compute a threshold"
            )
        precision, recall, thresholds = precision_recall_curve(targets, errors)
        # recall est décroissant dans precision_recall_curve
        # On cherche le plus grand seuil où recall >= target_recall
        valid = recall[:-1] >= target_recall
        if not np.any(valid):
            # Impossible d'atteindre le recall cible, prendre le seuil le plus bas
            return float(thresholds[0]), float(recall[0])
        idx = np.where(valid)[0][-1]
        return float(thresholds[idx]), float(recall[idx])

    def evaluate(
        self, errors: np.ndarray, targets: np.ndarray, threshold: float
    ) -> dict[str, float]:
        preds = (errors > threshold).astype(int)
        labels = [0, 1]
        report = classification_report(
            targets,
            preds,
            labels=labels,
            target_names=["Normal", "Anomalie"],
            zero_division=0,
        )

        unique_targets = np.unique(targets)
        if unique_targets.size < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(targets, errors)

        cm = confusion_matrix(targets, preds, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        return {
            "classification_report": report,
            "auc": auc,
            "confusion_matrix": cm,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        }
