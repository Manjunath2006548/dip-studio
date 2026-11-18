"""simulation.py -- illumination simulation and condition detection.

The Day image is a real photograph.  Midday, Evening and Night are produced
from it by working in linear light (gamma 2.2), applying an exposure gain, a
colour-temperature tint, a graded sun-glare term (midday) or a localised lamp
pool (night), clipping to the sensor range, gamma-encoding again and finally
adding signal-dependent sensor noise that grows as the scene gets darker.

The same model is used in reverse: when a user supplies a photograph that is
already in one of the four conditions, :func:`detect_condition` identifies
which one it is and :func:`invert_illumination` recovers a day-like base from
it.  The other three conditions are then generated from that base, so any
single uploaded photograph can anchor the full four-condition experiment.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import dip_core as dc

GAMMA = 2.2

# Exposure gains quoted in Section 2.1 of the report.
GAINS = {"midday": 2.6, "evening": 0.20, "night": 0.012}

# Fixed "look" of the simulated conditions.
TINTS = {
    "midday": (1.05, 1.00, 0.92),
    "evening": (1.20, 0.98, 0.76),
    "night": (0.82, 0.94, 1.20),
}
GLARE = {"midday": 0.12}          # graded sun glare strength
LAMP_AMP = {"night": 0.62}        # artificial light pool amplitude
LAMP_SPREAD = 0.20                # pool radius (squared, normalised coords)
LAMP_CENTRE = (0.32, 0.50)        # (x, y) as fractions of the frame
NOISE = {                          # (read-noise sigma, shot-noise coefficient)
    "day": (0.0, 0.0),
    "midday": (0.0012, 0.0006),
    "evening": (0.0010, 0.0008),
    "night": (0.0020, 0.0012),
}

CONDITION_LABEL = {
    "day": "Day",
    "midday": "Midday",
    "evening": "Evening",
    "night": "Night",
}


# ---------------------------------------------------------------------------
# shared lighting geometry
# ---------------------------------------------------------------------------
def _coords(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    return xx / max(w - 1, 1), yy / max(h - 1, 1)


def _glare(h: int, w: int) -> np.ndarray:
    xn, yn = _coords(h, w)
    return GLARE["midday"] * np.clip(
        1.0 - 0.85 * (0.65 * xn + 0.35 * yn), 0.0, 1.0) ** 1.4


def _lamp(h: int, w: int) -> np.ndarray:
    xn, yn = _coords(h, w)
    cx, cy = LAMP_CENTRE
    d2 = ((xn - cx) * 1.30) ** 2 + ((yn - cy) * 1.00) ** 2
    return np.exp(-d2 / LAMP_SPREAD)


def _ambient(condition: str, gain: float) -> np.ndarray:
    return gain * np.array(TINTS[condition], dtype=np.float64)


# ---------------------------------------------------------------------------
# forward simulation
# ---------------------------------------------------------------------------
def simulate_illumination(rgb: np.ndarray, condition: str, gain: float | None = None,
                          seed: int = 11) -> np.ndarray:
    """Return ``rgb`` re-lit as day / midday / evening / night (uint8 RGB)."""
    condition = condition.lower()
    if condition == "day":
        return rgb.copy()
    if condition not in GAINS:
        raise ValueError(f"unknown condition: {condition}")

    g = GAINS[condition] if gain is None else float(gain)
    rng = np.random.default_rng(seed)

    base = (rgb.astype(np.float64) / 255.0) ** GAMMA        # linear light
    h, w = base.shape[:2]

    x = base * _ambient(condition, g)

    if condition == "midday":
        x = x + _glare(h, w)[..., None] * np.array([1.00, 0.97, 0.88])

    if condition == "night":
        # Localised artificial light pool (street lamp).  The ambient term is
        # cool and the lamp term is warm; both multiply the scene reflectance,
        # so texture stays visible inside the pool.
        lamp = LAMP_AMP["night"] * _lamp(h, w)[..., None] * np.array(
            [1.00, 0.82, 0.58])
        x = base * (_ambient(condition, g) + lamp)

    x = np.clip(x, 0.0, 1.0)

    # Sensor noise in linear light: read noise + Poisson-like shot noise.
    read, shot = NOISE[condition]
    if read > 0 or shot > 0:
        x = x + rng.normal(0.0, np.sqrt(np.clip(x, 0.0, None) * shot))
        x = x + rng.normal(0.0, read, x.shape)
    x = np.clip(x, 0.0, 1.0)

    return np.round(255.0 * x ** (1.0 / GAMMA)).astype(np.uint8)


# ---------------------------------------------------------------------------
# inverse simulation: recover a day-like base from any condition
# ---------------------------------------------------------------------------
def invert_illumination(rgb: np.ndarray, condition: str,
                        gain: float | None = None) -> np.ndarray:
    """Estimate the day photograph that would produce ``rgb`` in ``condition``.

    This is the analytic inverse of :func:`simulate_illumination`: the glare
    term is subtracted, the night lamp pool is divided out, and the exposure
    gain / tint are undone in linear light before re-encoding at gamma 2.2.
    """
    condition = condition.lower()
    if condition == "day":
        return rgb.copy()
    if condition not in GAINS:
        raise ValueError(f"unknown condition: {condition}")

    g = GAINS[condition] if gain is None else float(gain)
    x = (rgb.astype(np.float64) / 255.0) ** GAMMA
    h, w = x.shape[:2]

    if condition == "midday":
        x = x - _glare(h, w)[..., None] * np.array([1.00, 0.97, 0.88])
        x = x / _ambient(condition, g)
    elif condition == "night":
        lamp = LAMP_AMP["night"] * _lamp(h, w)[..., None] * np.array(
            [1.00, 0.82, 0.58])
        x = x / (_ambient(condition, g) + lamp)
    else:  # evening
        x = x / _ambient(condition, g)

    x = np.clip(x, 0.0, 1.0)
    return np.round(255.0 * x ** (1.0 / GAMMA)).astype(np.uint8)


# ---------------------------------------------------------------------------
# condition detection
# ---------------------------------------------------------------------------
# Feature prototypes measured from the report's four simulated conditions at
# 640 x 480.  They describe how each illumination looks, independent of the
# scene's subject matter.
_PROTOTYPES = {
    "day":     {"log_mean": math.log(98.0 + 8), "bright_fraction": 0.00, "dark_fraction": 0.08,
                "log_range": math.log(3.7), "log_noise": math.log(5.5 + 0.5), "warmth": 1.20},
    "midday":  {"log_mean": math.log(161.0 + 8), "bright_fraction": 0.11, "dark_fraction": 0.00,
                "log_range": math.log(2.6), "log_noise": math.log(7.3 + 0.5), "warmth": 1.20},
    "evening": {"log_mean": math.log(47.0 + 8), "bright_fraction": 0.00, "dark_fraction": 0.45,
                "log_range": math.log(3.8), "log_noise": math.log(4.1 + 0.5), "warmth": 1.48},
    "night":   {"log_mean": math.log(44.0 + 8), "bright_fraction": 0.00, "dark_fraction": 0.49,
                "log_range": math.log(9.3), "log_noise": math.log(5.1 + 0.5), "warmth": 1.52},
}

_WEIGHTS = {
    "log_mean": 4.0,
    "bright_fraction": 3.0,
    "dark_fraction": 2.0,
    "log_range": 2.0,
    "log_noise": 0.7,
    "warmth": 0.5,
}


def illumination_features(rgb: np.ndarray) -> dict[str, float]:
    """Measured cues used to decide which illumination a photograph shows."""
    gray = dc.to_gray(rgb)
    g = gray.astype(np.float64)
    small = cv2.resize(g, (8, 6), interpolation=cv2.INTER_AREA)
    r_mean = float(rgb[..., 0].mean())
    b_mean = float(rgb[..., 2].mean())
    return {
        "mean": round(float(g.mean()), 2),
        "std": round(float(g.std()), 2),
        "dark_fraction": round(float((gray <= 40).mean()), 4),
        "bright_fraction": round(float((gray >= 250).mean()), 4),
        "local_range": round(float(small.max() / max(small.min(), 1.0)), 3),
        "noise_sigma": round(float(dc.noise_sigma(gray)), 3),
        "warmth": round((r_mean + 1.0) / (b_mean + 1.0), 4),
        "log_mean": math.log(float(g.mean()) + 8.0),
        "log_range": math.log(float(small.max() / max(small.min(), 1.0))),
        "log_noise": math.log(float(dc.noise_sigma(gray)) + 0.5),
    }


def detect_condition(rgb: np.ndarray) -> dict:
    """Identify the illumination condition of an uploaded photograph.

    Returns the best-matching condition, a margin-based confidence, the
    measured cues and the score for every candidate condition.
    """
    features = illumination_features(rgb)
    scores: dict[str, float] = {}
    for name, proto in _PROTOTYPES.items():
        scores[name] = round(sum(
            _WEIGHTS[key] * abs(features[key] - proto[key]) for key in _WEIGHTS), 5)

    ranked = sorted(scores.items(), key=lambda item: item[1])
    best_name, best_score = ranked[0]
    second_score = ranked[1][1]
    margin = max(0.0, second_score - best_score)
    confidence = round(1.0 / (1.0 + math.exp(-2.5 * margin)), 3)

    return {
        "condition": best_name,
        "label": CONDITION_LABEL[best_name],
        "confidence": confidence,
        "features": {k: features[k] for k in
                     ("mean", "std", "dark_fraction", "bright_fraction",
                      "local_range", "noise_sigma", "warmth")},
        "scores": scores,
    }


def conditions_from_photo(rgb: np.ndarray, gain_overrides: dict[str, float] | None = None,
                          seed: int = 11) -> tuple[dict[str, np.ndarray], dict]:
    """Detect a photo's condition and build all four conditions from it.

    The uploaded photo itself is kept as the detected condition; the other
    three are simulated from a day-like estimate recovered from that photo.
    """
    detection = detect_condition(rgb)
    anchor = detection["condition"]
    gains = gain_overrides or {}
    base = invert_illumination(rgb, anchor, gains.get(anchor))

    out: dict[str, np.ndarray] = {}
    for i, cond in enumerate(("day", "midday", "evening", "night")):
        out[cond] = simulate_illumination(
            base, cond, gains.get(cond), seed=seed + i)
    out[anchor] = rgb.copy()          # analyse the user's actual photograph
    return out, detection
