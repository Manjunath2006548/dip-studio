"""dip_core.py -- user-defined image-processing functions.

Every function in this module is written by hand with NumPy/OpenCV, exactly as
described in the project report.  OpenCV's cv2.equalizeHist is used only as an
independent cross-check of ``equalize``; cv2.createCLAHE supplies the CLAHE
comparison image.

Author: MANJUNATH BHASKAR HEBBAR (20241EVL0037)
"""

from __future__ import annotations

import cv2
import numpy as np

L = 256                       # number of grey levels (8-bit)
DEFAULT_SIZE = (640, 480)     # common working size, width x height


# ---------------------------------------------------------------------------
# 1. Loading and pre-processing
# ---------------------------------------------------------------------------
def preprocess_bgr(bgr: np.ndarray, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Centre-crop to the target aspect ratio, then resize with INTER_AREA."""
    if bgr is None:
        raise ValueError("image could not be decoded")
    h, w = bgr.shape[:2]
    cw = min(w, int(h * size[0] / size[1]))
    ch = min(h, int(w * size[1] / size[0]))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    bgr = bgr[y0:y0 + ch, x0:x0 + cw]
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return bgr


def load_and_preprocess(path: str, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Read a JPEG/PNG file and return an RGB image of the common size."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(preprocess_bgr(bgr, size), cv2.COLOR_BGR2RGB)


def decode_and_preprocess(data: bytes, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Decode an uploaded image from memory and return an RGB array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("unsupported or corrupt image file")
    return cv2.cvtColor(preprocess_bgr(bgr, size), cv2.COLOR_BGR2RGB)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma: I = 0.299 R + 0.587 G + 0.114 B."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Histogram, PDF and CDF
# ---------------------------------------------------------------------------
def histogram(gray: np.ndarray) -> np.ndarray:
    """h(r_k) = n_k for k = 0 .. 255."""
    return np.bincount(gray.ravel(), minlength=L).astype(np.int64)


def pdf_cdf(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised histogram (PDF) and its running sum (CDF)."""
    pdf = hist / hist.sum()
    return pdf, np.cumsum(pdf)


# ---------------------------------------------------------------------------
# 3. Histogram equalization (user-defined)
# ---------------------------------------------------------------------------
def equalize(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Global histogram equalization using Eq. (5) of the report.

    Returns the equalized uint8 image and the 256-entry mapping s = T(r).
    """
    h = histogram(gray)
    _, cdf = pdf_cdf(h)
    cdf_min = cdf[np.nonzero(h)[0][0]]
    denom = 1.0 - cdf_min
    if denom <= 0:                                   # constant image: no change
        T = np.arange(L, dtype=np.float64)
    else:
        T = np.round((cdf - cdf_min) / denom * (L - 1))
    T = np.clip(T, 0, L - 1).astype(np.uint8)
    return T[gray], T


def stretch(gray: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Linear contrast stretching: map the [lo, hi] percentile band to 0..255."""
    a, b = np.percentile(gray, [lo, hi])
    return np.clip((gray - a) / max(b - a, 1) * 255, 0, 255).astype(np.uint8)


def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization (OpenCV reference)."""
    tile = max(2, int(tile))
    return cv2.createCLAHE(clipLimit=float(clip_limit),
                           tileGridSize=(tile, tile)).apply(gray)


# ---------------------------------------------------------------------------
# 4. Quality parameters
# ---------------------------------------------------------------------------
def entropy(gray: np.ndarray) -> float:
    """Shannon entropy in bits per pixel."""
    p = histogram(gray) / gray.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise-variance estimate (sigma_n in grey levels)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    c = cv2.filter2D(gray.astype(np.float64), -1, k)[1:-1, 1:-1]
    h, w = gray.shape
    return float(np.sqrt(np.pi / 2) / (6.0 * (w - 2) * (h - 2)) * np.abs(c).sum())


def metrics(gray: np.ndarray, ref: np.ndarray | None = None) -> dict[str, float]:
    """All quality parameters of one image (optionally against a reference)."""
    g = gray.astype(np.float64)
    mean, std = g.mean(), g.std()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    m: dict[str, float] = {
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
        "mean": float(mean),
        "contrast_std": float(std),
        "michelson": float((g.max() - g.min()) / (g.max() + g.min() + 1e-9)),
        "dynamic_range": int(g.max() - g.min()),
        "levels_used": int((histogram(gray) > 0).sum()),
        "entropy": entropy(gray),
        "snr_db": float(20 * np.log10(mean / std)) if std > 0 else float("inf"),
        "sharpness": float(np.hypot(gx, gy).mean()),
        "noise_sigma": noise_sigma(gray),
    }
    m["snr_noise_db"] = (float(20 * np.log10(std / m["noise_sigma"]))
                         if m["noise_sigma"] > 0 else float("inf"))
    if ref is not None:
        mse = float(np.mean((g - ref.astype(np.float64)) ** 2))
        m["mse"] = mse
        m["psnr_db"] = float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else float("inf")
        m["ambe"] = float(abs(mean - ref.astype(np.float64).mean()))
    return m


# ---------------------------------------------------------------------------
# 5. One complete condition
# ---------------------------------------------------------------------------
def analyze_condition(rgb: np.ndarray, clip_limit: float = 2.0, tile: int = 8,
                      lo: float = 1, hi: float = 99) -> dict:
    """Run grayscale conversion, HE, OpenCV HE, CLAHE and stretching."""
    gray = to_gray(rgb)
    he, T = equalize(gray)
    ref_cv = cv2.equalizeHist(gray)
    clahe_img = clahe(gray, clip_limit, tile)
    st = stretch(gray, lo, hi)

    hist = histogram(gray)
    pdf, cdf = pdf_cdf(hist)
    he_hist = histogram(he)
    he_pdf, he_cdf = pdf_cdf(he_hist)

    return {
        "rgb": rgb,
        "gray": gray,
        "he": he,
        "T": T,
        "clahe": clahe_img,
        "stretch": st,
        "hist": hist,
        "pdf": pdf,
        "cdf": cdf,
        "he_hist": he_hist,
        "he_pdf": he_pdf,
        "he_cdf": he_cdf,
        "max_diff_cv": int(np.abs(he.astype(int) - ref_cv.astype(int)).max()),
        "metrics": {
            "original": metrics(gray),
            "he": metrics(he, gray),
            "clahe": metrics(clahe_img, gray),
            "stretch": metrics(st, gray),
        },
    }
