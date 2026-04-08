"""Genere les diagrammes d'architecture RNN-VAE et CNN-LSTM pour l'affiche CANlock."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath
from pathlib import Path

NAVY = "#1B2A4A"
BLUE = "#2E5C9A"
CYAN = "#5BC0DE"
DARK_TEAL = "#1A5276"
WHITE = "#FFFFFF"
BORDER_BLUE = "#2E5C9A"
ACCENT_GREEN = "#27AE60"

OUT_DIR = Path("doc/images/poster")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Espacement fleche standard
GAP = 0.35


def draw_box(ax, x, y, w, h, text, color=NAVY, text_color=WHITE, fontsize=9):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=NAVY, linewidth=1.5, zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4,
            linespacing=1.15)
    return x - w/2, x + w/2  # left, right edges


def draw_trapezoid(ax, x, y, w_top, w_bot, h, text, color=NAVY, text_color=WHITE, fontsize=8):
    verts = [
        (x - w_bot/2, y - h/2), (x + w_bot/2, y - h/2),
        (x + w_top/2, y + h/2), (x - w_top/2, y + h/2),
        (x - w_bot/2, y - h/2),
    ]
    ax.add_patch(plt.Polygon(verts, facecolor=color, edgecolor=NAVY, linewidth=1.5, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4, linespacing=1.15)
    hw = max(w_top, w_bot) / 2
    return x - hw, x + hw


def draw_diamond(ax, x, y, w, h, text, color=CYAN, text_color=NAVY, fontsize=10):
    verts = [[x, y + h/2], [x + w/2, y], [x, y - h/2], [x - w/2, y]]
    ax.add_patch(plt.Polygon(verts, facecolor=color, edgecolor=NAVY, linewidth=1.5, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=CYAN, lw=2.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16),
                zorder=2)


def draw_label(ax, x, y, text, color=CYAN, fontsize=9):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontstyle="italic", zorder=5)


def draw_brace(ax, x1, x2, y, height=0.4, text="", color=CYAN, fontsize=11):
    mid = (x1 + x2) / 2
    q = min(0.3, (x2 - x1) * 0.08)
    verts = [
        (x1, y), (x1, y + height * 0.5),
        (x1 + q, y + height * 0.6), (mid - q, y + height * 0.6),
        (mid, y + height),
        (mid + q, y + height * 0.6), (x2 - q, y + height * 0.6),
        (x2, y + height * 0.5), (x2, y),
    ]
    codes = [MPath.MOVETO] + [MPath.CURVE3, MPath.CURVE3] * 4
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor="none",
                           edgecolor=color, lw=2.5, zorder=2))
    if text:
        ax.text(mid, y + height + 0.15, text, ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color=color, zorder=5)


def add_border(fig):
    fig.patches.append(plt.Rectangle(
        (0, 0), 1, 1, transform=fig.transFigure,
        fill=False, edgecolor=BORDER_BLUE, linewidth=3, zorder=100
    ))


# =========================================================================
#  RNN-VAE
# =========================================================================
def fig_rnn_vae():
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor(WHITE); ax.set_facecolor(WHITE); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(-2.5, 5.8); ax.set_aspect("equal")

    ax.text(7, 5.5, "RNN-VAE (Variational Auto-Encoder)",
            ha="center", fontsize=20, fontweight="bold", color=NAVY)

    Y = 2.5
    BH = 1.15
    BW = 1.4

    # --- Positions (evenly spaced) ---
    # X=0.5 label | 1.6 enc_lstm | 3.4 dropout | 5.0 mu/logvar | 6.6 Z |
    #             | 8.2 linear   | 9.8 dec_lstm | 11.4 linear_out | 12.8 X_hat

    # X input
    draw_label(ax, 0.5, Y, "X", color=NAVY, fontsize=14)
    draw_arrow(ax, 0.8, Y, 1.0 - 0.05, Y)

    # Encodeur Bi-LSTM
    draw_trapezoid(ax, 1.8, Y, 1.1, 1.6, BH,
                   "Encodeur\nBi-LSTM\n(64 hidden)", color=NAVY, fontsize=8)
    draw_arrow(ax, 2.6, Y, 2.6 + GAP, Y)

    # Dropout
    draw_box(ax, 3.5, Y, 0.9, 0.7, "Dropout\n0.2", color=DARK_TEAL, fontsize=8)
    draw_arrow(ax, 3.95, Y + 0.15, 4.3, Y + 0.75)
    draw_arrow(ax, 3.95, Y - 0.15, 4.3, Y - 0.75)

    # fc_mu / fc_logvar
    draw_box(ax, 4.8, Y + 0.9, 0.8, 0.55, "fc_mu", color=BLUE, fontsize=8)
    draw_box(ax, 4.8, Y - 0.9, 0.8, 0.55, "fc_logvar", color=BLUE, fontsize=8)
    draw_label(ax, 5.4, Y + 0.9, "mu", fontsize=9)
    draw_label(ax, 5.4, Y - 0.9, "sigma", fontsize=9)

    # Z latent
    draw_arrow(ax, 5.2, Y + 0.65, 5.7, Y + 0.2)
    draw_arrow(ax, 5.2, Y - 0.65, 5.7, Y - 0.2)
    draw_box(ax, 6.2, Y, 1.0, 1.0, "Z\nlatent\n(32)", color=CYAN, text_color=NAVY, fontsize=9)
    draw_label(ax, 6.2, Y - 0.85, "Echantillonne", fontsize=7)
    draw_arrow(ax, 6.7, Y, 6.7 + GAP, Y)

    # Linear expand
    draw_box(ax, 7.6, Y, 1.0, 0.8, "Linear\n32 -> 128", color=BLUE, fontsize=8)
    draw_label(ax, 7.6, Y - 0.7, "repeat seq_len", fontsize=7)
    draw_arrow(ax, 8.1, Y, 8.1 + GAP, Y)

    # Decodeur Bi-LSTM
    draw_trapezoid(ax, 9.3, Y, 1.6, 1.1, BH,
                   "Decodeur\nBi-LSTM\n(64 hidden)", color=NAVY, fontsize=8)
    draw_arrow(ax, 10.1, Y, 10.1 + GAP, Y)

    # Dropout dec
    draw_box(ax, 11.0, Y, 0.9, 0.7, "Dropout\n0.2", color=DARK_TEAL, fontsize=8)
    draw_arrow(ax, 11.45, Y, 11.45 + GAP, Y)

    # Output linear
    draw_box(ax, 12.3, Y, 0.9, 0.7, "Linear\n-> 11", color=BLUE, fontsize=8)
    draw_arrow(ax, 12.75, Y, 12.75 + GAP, Y)

    # X_hat
    draw_label(ax, 13.5, Y, "X_hat", color=NAVY, fontsize=13)

    # --- Accolades ---
    brace_y = 3.35
    draw_brace(ax, 0.9, 5.3, brace_y, height=0.45, text="ENCODEUR", fontsize=12)
    draw_brace(ax, 7.1, 12.8, brace_y, height=0.45, text="DECODEUR", fontsize=12)

    # --- Erreur de reconstruction ---
    err_y = 0.3
    draw_arrow(ax, 0.5, Y - 0.8, 0.5, err_y)
    draw_arrow(ax, 0.5, err_y, 4.8, err_y)
    draw_arrow(ax, 13.5, Y - 0.6, 13.5, err_y)
    draw_arrow(ax, 13.5, err_y, 9.2, err_y)

    draw_box(ax, 7.0, err_y, 2.6, 0.7,
             "Erreur de reconstruction (MSE)", color=DARK_TEAL, fontsize=9)

    # Fleche vers anomalie
    draw_arrow(ax, 7.0, err_y - 0.4, 7.0, err_y - 0.9)

    # Losange
    draw_diamond(ax, 7.0, err_y - 1.5, 1.8, 0.9,
                 "Anomalie ?", fontsize=11)

    add_border(fig)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT_DIR / "architecture_rnn_vae.png", dpi=250, bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"[+] {OUT_DIR / 'architecture_rnn_vae.png'}")


# =========================================================================
#  CNN-LSTM
# =========================================================================
def fig_cnn_lstm():
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor(WHITE); ax.set_facecolor(WHITE); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(-2.5, 5.8); ax.set_aspect("equal")

    ax.text(7, 5.5, "CNN-LSTM Autoencoder",
            ha="center", fontsize=20, fontweight="bold", color=NAVY)

    Y = 2.5
    BH = 1.1
    BW_S = 0.9   # small box
    BW_M = 1.15  # medium box
    BW_L = 1.3   # large box

    # --- Positions regulieres ---
    # 0.4 X | 1.3 conv1 | 2.5 pool | 3.5 conv2 | 4.7 pool | 5.8 bilstm_enc |
    #       | 7.1 bottleneck | 8.3 bilstm_dec | 9.7 convT | 11.0 conv_out | 12.2 interp | 13.2 X_hat

    # X input
    draw_label(ax, 0.4, Y, "X", color=NAVY, fontsize=14)
    draw_arrow(ax, 0.65, Y, 0.65 + GAP, Y)

    # Conv1d #1
    draw_box(ax, 1.5, Y, BW_M, BH, "Conv1d\n11->32\nBN+ReLU", color=BLUE, fontsize=8)
    draw_arrow(ax, 2.07, Y, 2.07 + GAP, Y)

    # Pool /2
    draw_box(ax, 2.8, Y, 0.6, 0.6, "Pool\n/2", color=DARK_TEAL, fontsize=7)
    draw_arrow(ax, 3.1, Y, 3.1 + GAP, Y)

    # Conv1d #2
    draw_box(ax, 3.9, Y, BW_M, BH, "Conv1d\n32->64\nBN+ReLU", color=BLUE, fontsize=8)
    draw_arrow(ax, 4.47, Y, 4.47 + GAP, Y)

    # Pool /2
    draw_box(ax, 5.2, Y, 0.6, 0.6, "Pool\n/2", color=DARK_TEAL, fontsize=7)
    draw_arrow(ax, 5.5, Y, 5.5 + GAP, Y)

    # Bi-LSTM enc
    draw_trapezoid(ax, 6.4, Y, 0.9, 1.3, BH,
                   "Bi-LSTM\n32 hid\nDropout", color=NAVY, fontsize=8)
    draw_arrow(ax, 7.05, Y, 7.05 + GAP, Y)

    # Bottleneck
    draw_box(ax, 7.9, Y, BW_S, 0.8, "Linear\n64->32", color=CYAN, text_color=NAVY, fontsize=8)
    draw_label(ax, 7.9, Y - 0.7, "Bottleneck", fontsize=7)
    draw_arrow(ax, 8.35, Y, 8.35 + GAP, Y)

    # Bi-LSTM dec
    draw_trapezoid(ax, 9.3, Y, 1.3, 0.9, BH,
                   "Bi-LSTM\n64 hid", color=NAVY, fontsize=8)
    draw_arrow(ax, 9.95, Y, 9.95 + GAP, Y)

    # ConvTranspose1d
    draw_box(ax, 10.8, Y, BW_L, BH, "ConvT1d\n128->64\n64->32", color=BLUE, fontsize=8)
    draw_arrow(ax, 11.45, Y, 11.45 + GAP, Y)

    # Conv1d final
    draw_box(ax, 12.3, Y, BW_S, 0.8, "Conv1d\n32->11", color=BLUE, fontsize=8)
    draw_arrow(ax, 12.75, Y, 12.75 + GAP, Y)

    # X_hat
    draw_label(ax, 13.4, Y, "X_hat", color=NAVY, fontsize=13)

    # --- Accolades ---
    brace_y = 3.3
    draw_brace(ax, 0.9, 5.55, brace_y, height=0.45, text="ENCODEUR CNN", fontsize=11)
    draw_brace(ax, 5.7, 8.4, brace_y, height=0.45, text="LSTM", fontsize=11)
    draw_brace(ax, 8.6, 12.8, brace_y, height=0.45, text="DECODEUR", fontsize=11)

    # --- Erreur de reconstruction ---
    err_y = 0.3
    draw_arrow(ax, 0.4, Y - 0.7, 0.4, err_y)
    draw_arrow(ax, 0.4, err_y, 4.8, err_y)
    draw_arrow(ax, 13.4, Y - 0.6, 13.4, err_y)
    draw_arrow(ax, 13.4, err_y, 9.2, err_y)

    draw_box(ax, 7.0, err_y, 2.6, 0.7,
             "Erreur de reconstruction (MSE)", color=DARK_TEAL, fontsize=9)

    draw_arrow(ax, 7.0, err_y - 0.4, 7.0, err_y - 0.9)

    draw_diamond(ax, 7.0, err_y - 1.5, 1.8, 0.9,
                 "Anomalie ?", fontsize=11)

    add_border(fig)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT_DIR / "architecture_cnn_lstm.png", dpi=250, bbox_inches="tight",
                facecolor=WHITE)
    plt.close()
    print(f"[+] {OUT_DIR / 'architecture_cnn_lstm.png'}")


if __name__ == "__main__":
    fig_rnn_vae()
    fig_cnn_lstm()
    print("\n[+] Diagrammes d'architecture generes dans doc/images/poster/")
