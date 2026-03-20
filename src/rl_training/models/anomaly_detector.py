import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, confusion_matrix

class AnomalyDetector:
    """
    Classe utilitaire pour la détection d'anomalies par reconstruction.
    Gère le calcul des erreurs, le choix du seuil optimal et l'évaluation.
    """
    def __init__(self, model, datamodule, device='cpu'):
        self.model = model
        self.datamodule = datamodule
        self.device = device

    def compute_reconstruction_errors(self):
        self.model.eval()
        self.model.to(self.device)
        reconstruction_errors = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in self.datamodule.test_dataloader():
                batch_x = batch_x.to(self.device)
                x_hat = self.model(batch_x)
                mse = torch.mean((batch_x - x_hat) ** 2, dim=(1, 2))
                reconstruction_errors.extend(mse.cpu().numpy())
                all_targets.extend(batch_y.numpy())
        return np.array(reconstruction_errors), np.array(all_targets)

    def find_optimal_threshold(self, errors, targets):
        precision, recall, thresholds = precision_recall_curve(targets, errors)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        return optimal_threshold, f1_scores[optimal_idx]

    def evaluate(self, errors, targets, threshold):
        preds = (errors > threshold).astype(int)
        report = classification_report(targets, preds, target_names=['Normal', 'Anomalie'])
        auc = roc_auc_score(targets, errors)
        cm = confusion_matrix(targets, preds)
        tn, fp, fn, tp = cm.ravel()
        return {
            'classification_report': report,
            'auc': auc,
            'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
