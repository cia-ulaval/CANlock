"""Génère les figures pour l'affiche CANlock — style cohérent avec l'affiche."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Style affiche : fond blanc, bleu marine (#1B2A4A), bordures bleues
NAVY = "#1B2A4A"
BLUE = "#2E5C9A"
LIGHT_BLUE = "#4A90D9"
ACCENT_GREEN = "#27AE60"
ACCENT_RED = "#C0392B"
LIGHT_BG = "#F5F7FA"
BORDER_BLUE = "#2E5C9A"

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 13
matplotlib.rcParams["axes.edgecolor"] = BORDER_BLUE
matplotlib.rcParams["axes.linewidth"] = 1.5
matplotlib.rcParams["figure.facecolor"] = "white"

OUT_DIR = Path("doc/images/poster")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_border(fig, color=BORDER_BLUE, linewidth=3):
    """Ajoute une bordure bleue autour de la figure."""
    fig.patches.append(plt.Rectangle(
        (0, 0), 1, 1, transform=fig.transFigure,
        fill=False, edgecolor=color, linewidth=linewidth, zorder=100
    ))


def fig1_comparison_table():
    """CNN-LSTM vs RNN-VAE — tableau comparatif."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    fig.set_facecolor("white")

    metrics = ["ROC-AUC", "FPR", "Recall", "F1-Score", "Accuracy"]
    rnn_vae = [0.990, 0.294, 0.995, 0.940, 0.91]
    cnn_lstm = [0.998, 0.062, 0.995, 0.988, 0.98]

    cell_text = []
    for m, r, c in zip(metrics, rnn_vae, cnn_lstm):
        cell_text.append([m, f"{r:.3f}", f"{c:.3f}"])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Metrique", "RNN-VAE", "CNN-LSTM"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.0)

    # Header style affiche
    for j in range(3):
        table[0, j].set_facecolor(NAVY)
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=14)
        table[0, j].set_edgecolor(BORDER_BLUE)

    for i in range(1, len(metrics) + 1):
        # Colonne metrique
        table[i, 0].set_facecolor(LIGHT_BG)
        table[i, 0].set_text_props(fontweight="bold", color=NAVY)
        table[i, 0].set_edgecolor(BORDER_BLUE)
        # RNN-VAE
        table[i, 1].set_facecolor("#FDE8E8")
        table[i, 1].set_text_props(color=ACCENT_RED)
        table[i, 1].set_edgecolor(BORDER_BLUE)
        # CNN-LSTM
        table[i, 2].set_facecolor("#E8F8F0")
        table[i, 2].set_text_props(fontweight="bold", color=ACCENT_GREEN)
        table[i, 2].set_edgecolor(BORDER_BLUE)

    ax.set_title(
        "CNN-LSTM vs RNN-VAE", fontsize=18, fontweight="bold", color=NAVY, pad=25
    )
    add_border(fig)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / "cnn_lstm_vs_rnn_vae.png", dpi=250, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[+] {OUT_DIR / 'cnn_lstm_vs_rnn_vae.png'}")


def fig2_evolution_bar():
    """Evolution des performances CNN-LSTM."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    steps = [
        "SPN\n50k",
        "Payloads\nbruts",
        "+CAN ID\n+temps",
        "200k\nmessages",
        "500k\nstride=10",
        "Taux\nattaque x3",
        "Final\n(optimise)",
    ]
    roc_auc = [0.52, 0.54, 0.67, 0.70, 0.92, 0.996, 0.998]
    recall = [0.997, 0.997, 0.997, 0.917, 0.917, 0.979, 0.995]
    fpr = [1.00, 1.00, 0.99, 0.68, 0.29, 0.037, 0.062]

    x = np.arange(len(steps))
    width = 0.25

    bars1 = ax.bar(x - width, roc_auc, width, label="ROC-AUC", color=BLUE, edgecolor=NAVY, linewidth=0.5)
    bars2 = ax.bar(x, recall, width, label="Recall", color=ACCENT_GREEN, edgecolor=NAVY, linewidth=0.5)
    bars3 = ax.bar(x + width, fpr, width, label="FPR", color=ACCENT_RED, edgecolor=NAVY, linewidth=0.5)

    ax.set_ylabel("Score", fontsize=14, color=NAVY, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=10, color=NAVY)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors=NAVY)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9, edgecolor=BORDER_BLUE)
    ax.set_title(
        "Evolution des performances CNN-LSTM", fontsize=18, fontweight="bold", color=NAVY
    )
    ax.axhline(y=0.99, color=NAVY, linestyle="--", alpha=0.3, linewidth=1)
    ax.text(6.4, 1.0, "Recall cible (0.99)", fontsize=9, color=NAVY, alpha=0.5, ha="right")

    # Valeurs finales
    for bar in bars1[-1:]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=BLUE)
    for bar in bars2[-1:]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=ACCENT_GREEN)
    for bar in bars3[-1:]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=ACCENT_RED)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_BLUE)
    ax.spines["bottom"].set_color(BORDER_BLUE)

    add_border(fig)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / "evolution_performances.png", dpi=250, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[+] {OUT_DIR / 'evolution_performances.png'}")


def fig3_confusion_matrix():
    """Matrices de confusion CNN-LSTM."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.set_facecolor("white")

    def draw_cm(ax, cm, cmap_base, title):
        from matplotlib.colors import LinearSegmentedColormap
        if cmap_base == "blue":
            colors = ["#FFFFFF", "#D6E4F0", "#4A90D9", NAVY]
        else:
            colors = ["#FFFFFF", "#D4EFDF", "#27AE60", "#1A7A3E"]
        cmap = LinearSegmentedColormap.from_list("custom", colors)

        im = ax.imshow(cm, cmap=cmap, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomalie"], fontsize=12, color=NAVY)
        ax.set_yticklabels(["Normal", "Anomalie"], fontsize=12, color=NAVY)
        ax.set_xlabel("Prediction", fontsize=13, color=NAVY, fontweight="bold")
        ax.set_ylabel("Reel", fontsize=13, color=NAVY, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=10)
        ax.tick_params(colors=NAVY)

        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_BLUE)
            spine.set_linewidth(1.5)

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > 2000 else NAVY
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)

    cm1 = np.array([[1559, 42], [45, 4122]])
    draw_cm(ax1, cm1, "blue", "Seuil F1-optimal\n(Recall = 98.9%  |  FPR = 2.6%)")

    cm2 = np.array([[1502, 99], [21, 4146]])
    draw_cm(ax2, cm2, "green", "Seuil Recall >= 99.5%\n(Recall = 99.5%  |  FPR = 6.2%)")

    fig.suptitle("Matrices de confusion - CNN-LSTM",
                 fontsize=17, fontweight="bold", color=NAVY)
    add_border(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=250, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[+] {OUT_DIR / 'confusion_matrices.png'}")


def fig4_features():
    """Features extraites des messages CAN bruts."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    fig.set_facecolor("white")

    features = [
        ("CAN ID", "Masquerade, DDoS", BLUE),
        ("byte_0 .. byte_7  (payload)", "Spoofing", LIGHT_BLUE),
        ("delta_t  (inter-message)", "Replay, Suspension, DDoS", BLUE),
        ("freq  (fenetre glissante = 50)", "DDoS, Suspension", LIGHT_BLUE),
    ]

    cell_text = [[f, a] for f, a, _ in features]

    table = ax.table(
        cellText=cell_text,
        colLabels=["Feature (11 total)", "Attaques detectees"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.2)

    for j in range(2):
        table[0, j].set_facecolor(NAVY)
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=14)
        table[0, j].set_edgecolor(BORDER_BLUE)

    for i, (_, _, c) in enumerate(features, 1):
        table[i, 0].set_facecolor(c + "18")
        table[i, 0].set_text_props(fontweight="bold", color=NAVY)
        table[i, 0].set_edgecolor(BORDER_BLUE)
        table[i, 1].set_facecolor(LIGHT_BG)
        table[i, 1].set_text_props(color=NAVY)
        table[i, 1].set_edgecolor(BORDER_BLUE)

    ax.set_title(
        "Features extraites des messages CAN bruts",
        fontsize=17, fontweight="bold", color=NAVY, pad=25,
    )
    add_border(fig)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT_DIR / "features_extraction.png", dpi=250, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[+] {OUT_DIR / 'features_extraction.png'}")


def fig5_confusion_matrix_rnn_vae():
    """Matrices de confusion RNN-VAE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.set_facecolor("white")

    def draw_cm(ax, cm, cmap_base, title):
        from matplotlib.colors import LinearSegmentedColormap
        if cmap_base == "blue":
            colors = ["#FFFFFF", "#D6E4F0", "#4A90D9", NAVY]
        else:
            colors = ["#FFFFFF", "#D4EFDF", "#27AE60", "#1A7A3E"]
        cmap = LinearSegmentedColormap.from_list("custom", colors)

        im = ax.imshow(cm, cmap=cmap, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomalie"], fontsize=12, color=NAVY)
        ax.set_yticklabels(["Normal", "Anomalie"], fontsize=12, color=NAVY)
        ax.set_xlabel("Prediction", fontsize=13, color=NAVY, fontweight="bold")
        ax.set_ylabel("Reel", fontsize=13, color=NAVY, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=10)
        ax.tick_params(colors=NAVY)

        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_BLUE)
            spine.set_linewidth(1.5)

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > 2000 else NAVY
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)

    cm1 = np.array([[1486, 115], [134, 4033]])
    draw_cm(ax1, cm1, "blue", "Seuil F1-optimal\n(Recall = 96.8%  |  FPR = 7.2%)")

    cm2 = np.array([[1131, 470], [21, 4146]])
    draw_cm(ax2, cm2, "green", "Seuil Recall >= 99.5%\n(Recall = 99.5%  |  FPR = 29.4%)")

    fig.suptitle("Matrices de confusion - RNN-VAE",
                 fontsize=17, fontweight="bold", color=NAVY)
    add_border(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "confusion_matrices_rnn_vae.png", dpi=250, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[+] {OUT_DIR / 'confusion_matrices_rnn_vae.png'}")


if __name__ == "__main__":
    fig1_comparison_table()
    fig2_evolution_bar()
    fig3_confusion_matrix()
    fig4_features()
    fig5_confusion_matrix_rnn_vae()
    print("\n[+] Toutes les figures generees dans doc/images/poster/")
