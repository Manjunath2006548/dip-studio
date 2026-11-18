"""plotting.py -- Matplotlib figures for the DIP workbench.

All figures are produced server-side and returned to the browser as base64
data URLs, so the interface needs no third-party JavaScript libraries.

Rendering is kept fast: photos are displayed with ``interpolation="none"``,
histograms use a single ``stairs`` artist instead of 256 bars, the overview
grid is downsampled before display, and figures are encoded as JPEG unless
crisp text is essential.
"""

from __future__ import annotations

import base64
import io

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#c7d0dc",
    "axes.linewidth": 0.8,
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "xtick.color": "#445064",
    "ytick.color": "#445064",
    "text.color": "#1b2430",
    "axes.labelcolor": "#445064",
})

C_ORIG = "#3b82f6"
C_HE = "#0d9488"
C_CDF = "#f59e0b"
C_CLAHE = "#8b5cf6"
C_STRETCH = "#ef4444"


def _b64(fig, fmt: str = "jpg", dpi: int = 96, quality: int = 88) -> str:
    buf = io.BytesIO()
    kwargs = dict(format=fmt, dpi=dpi, bbox_inches="tight",
                  facecolor=fig.get_facecolor())
    if fmt in ("jpg", "jpeg"):
        kwargs["pil_kwargs"] = {"quality": quality}
    fig.savefig(buf, **kwargs)
    plt.close(fig)
    mime = "image/jpeg" if fmt in ("jpg", "jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _down(img: np.ndarray, width: int = 320) -> np.ndarray:
    """Small copy for multi-image grids (reduces rasterisation cost)."""
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, max(1, round(h * width / w))),
                      interpolation=cv2.INTER_AREA)


def _style_hist_axis(ax) -> None:
    ax.set_xlim(0, 255)
    ax.set_xlabel("grey level r")
    ax.set_ylabel("pixels")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def condition_panel(cond: str, d: dict) -> str:
    """2 x 3 figure: original colour, grayscale, HE, histogram+CDF, T(r), HE hist."""
    m0 = d["metrics"]["original"]
    m1 = d["metrics"]["he"]
    fig, axs = plt.subplots(2, 3, figsize=(12.8, 6.7), constrained_layout=True)
    fig.suptitle(
        f"{cond.title()} illumination   ·   mean {m0['mean']:.1f} → {m1['mean']:.1f}"
        f"   ·   contrast σ {m0['contrast_std']:.1f} → {m1['contrast_std']:.1f}"
        f"   ·   entropy {m0['entropy']:.2f} → {m1['entropy']:.2f} bits",
        fontsize=11.5, fontweight="bold", color="#0f172a")

    axs[0, 0].imshow(d["rgb"], interpolation="none")
    axs[0, 0].set_title("(a) original colour")
    axs[0, 1].imshow(d["gray"], cmap="gray", vmin=0, vmax=255, interpolation="none")
    axs[0, 1].set_title("(b) grayscale  I = 0.299R + 0.587G + 0.114B", fontsize=9)
    axs[0, 2].imshow(d["he"], cmap="gray", vmin=0, vmax=255, interpolation="none")
    axs[0, 2].set_title(f"(c) histogram equalized  (mean {m1['mean']:.1f})")
    for ax in axs[0]:
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axs[1, 0]
    ax.stairs(d["hist"], np.arange(257), fill=True, color=C_ORIG,
              alpha=0.82, linewidth=0)
    ax.set_title("(d) original histogram + CDF")
    _style_hist_axis(ax)
    ax2 = ax.twinx()
    ax2.plot(np.arange(256), d["cdf"], color=C_CDF, lw=1.6)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("CDF", color=C_CDF)
    ax2.tick_params(axis="y", colors=C_CDF)
    ax2.spines[["top"]].set_visible(False)

    ax = axs[1, 1]
    ax.plot(np.arange(256), d["T"], color=C_HE, lw=2.0, label="s = T(r)")
    ax.plot([0, 255], [0, 255], "--", color="#94a3b8", lw=1.0, label="identity")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("input r")
    ax.set_ylabel("output s")
    ax.set_title(f"(e) transformation s = T(r)  ·  max |HE − cv2| = {d['max_diff_cv']}",
                 fontsize=9)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axs[1, 2]
    ax.stairs(d["he_hist"], np.arange(257), fill=True, color=C_HE,
              alpha=0.85, linewidth=0)
    ax.set_title(f"(f) equalized histogram + CDF  ·  {m1['levels_used']} levels used")
    _style_hist_axis(ax)
    ax2 = ax.twinx()
    ax2.plot(np.arange(256), d["he_cdf"], color=C_CDF, lw=1.6)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("CDF", color=C_CDF)
    ax2.tick_params(axis="y", colors=C_CDF)
    ax2.spines[["top"]].set_visible(False)

    return _b64(fig, "jpg", 96, 88)


def overview_figure(data: dict) -> str:
    """Original vs global HE vs CLAHE vs contrast stretching for all conditions."""
    conds = list(data.keys())
    cols = [("gray", "Original grayscale"), ("he", "Global HE"),
            ("clahe", "CLAHE"), ("stretch", "Contrast stretch")]
    fig, axs = plt.subplots(len(conds), 4, figsize=(12.8, 2.45 * len(conds)),
                            constrained_layout=True, squeeze=False)
    for r, cond in enumerate(conds):
        for c, (key, label) in enumerate(cols):
            ax = axs[r, c]
            ax.imshow(_down(data[cond][key], 320), cmap="gray", vmin=0, vmax=255,
                      interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(cond.title(), fontsize=10, fontweight="bold")
        m = data[cond]["metrics"]
        axs[r, 0].set_xlabel(f"σ {m['original']['contrast_std']:.1f}", fontsize=8, labelpad=2)
        axs[r, 1].set_xlabel(f"σ {m['he']['contrast_std']:.1f}", fontsize=8, labelpad=2)
        axs[r, 2].set_xlabel(f"σ {m['clahe']['contrast_std']:.1f}", fontsize=8, labelpad=2)
        axs[r, 3].set_xlabel(f"σ {m['stretch']['contrast_std']:.1f}", fontsize=8, labelpad=2)
    fig.suptitle("Global HE vs CLAHE vs contrast stretching  (standard deviation σ shown)",
                 fontsize=12, fontweight="bold", color="#0f172a")
    return _b64(fig, "jpg", 94, 86)


def metrics_figure(results: dict) -> str:
    """Bar chart of mean, contrast, entropy and noise-based SNR, original vs HE."""
    conds = list(results.keys())
    specs = [("mean", "Mean intensity μ"),
             ("contrast_std", "Contrast σ (grey levels)"),
             ("entropy", "Entropy H (bits/pixel)"),
             ("snr_noise_db", "SNR (noise based, dB)")]
    x = np.arange(len(conds))
    fig, axs = plt.subplots(2, 2, figsize=(10.8, 6.4), constrained_layout=True)
    for ax, (key, label) in zip(axs.ravel(), specs):
        before = [results[c]["original"][key] for c in conds]
        after = [results[c]["he"][key] for c in conds]
        w = 0.36
        b1 = ax.bar(x - w / 2, before, w, color="#cbd5e1", edgecolor="#94a3b8",
                    linewidth=0.6, label="original")
        b2 = ax.bar(x + w / 2, after, w, color=C_HE, label="after global HE")
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in conds])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "entropy":
            ax.set_ylim(0, 8.2)
        ax.legend(frameon=False, fontsize=8, loc="best")
        for bars in (b1, b2):
            for rect in bars:
                h = rect.get_height()
                ax.annotate(f"{h:.1f}" if key != "entropy" else f"{h:.2f}",
                            (rect.get_x() + rect.get_width() / 2, h),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=7, color="#475569")
    fig.suptitle("Quality parameters before and after histogram equalization",
                 fontsize=12, fontweight="bold", color="#0f172a")
    return _b64(fig, "jpg", 96, 92)


def techniques_figure(results: dict) -> str:
    """Contrast / entropy / noise-SNR for original, HE, CLAHE and stretching."""
    conds = list(results.keys())
    techs = [("original", "Original", "#cbd5e1", "#94a3b8"),
             ("he", "Global HE", C_HE, C_HE),
             ("clahe", "CLAHE", C_CLAHE, C_CLAHE),
             ("stretch", "Stretch", C_STRETCH, C_STRETCH)]
    specs = [("contrast_std", "Contrast σ (grey levels)"),
             ("entropy", "Entropy H (bits/pixel)"),
             ("snr_noise_db", "SNR (noise based, dB)")]
    x = np.arange(len(conds))
    w = 0.20
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 4.1), constrained_layout=True)
    for ax, (key, label) in zip(axs, specs):
        for i, (tkey, tlabel, color, edge) in enumerate(techs):
            vals = [results[c][tkey][key] for c in conds]
            ax.bar(x + (i - 1.5) * w, vals, w, label=tlabel, color=color,
                   edgecolor=edge, linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in conds])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "entropy":
            ax.set_ylim(0, 8.2)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Technique comparison across the four illumination conditions",
                 fontsize=12, fontweight="bold", color="#0f172a")
    return _b64(fig, "jpg", 96, 92)
