"""Paired comparisons of two classifiers on one test set.

The study's question is whether a model representation adds anything to a non-model control. On a
fixed test set the two probes' per-window outcomes are paired, so the right tools are paired:

- ``mcnemar``: exact two-sided McNemar test on the discordant windows (superiority);
- ``paired_difference``: mean accuracy difference with a normal-approximation interval;
- ``tost``: two one-sided tests of equivalence within a margin (TOST, Schuirmann 1987), which is
  what licenses "no difference" -- a non-significant superiority test does not;
- ``mcnemar_sample_size``: test windows needed to detect a given difference;
- ``vinfo_bits`` and ``paired_logloss_difference``: predictive V-information in bits (Xu et al.,
  2020) estimated from held-out cross-entropy, and the paired difference of two probes' losses,
  which is the conditional V-information of conditional probing (Hewitt et al., 2021) when one probe
  sees the control and the other sees the control and the representation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

LOG2 = math.log(2.0)


def mcnemar(correct_a: NDArray, correct_b: NDArray) -> dict[str, float]:
    """Exact two-sided McNemar test of equal accuracy for paired 0/1 outcomes."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must have the same shape")
    n01 = int(np.sum(~a & b))  # b right, a wrong
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return {"n_a_only": n10, "n_b_only": n01, "p_value": p}


def paired_difference(correct_a: NDArray, correct_b: NDArray, level: float = 0.95) -> dict:
    """Accuracy of a minus accuracy of b with a normal-approximation paired interval."""
    d = np.asarray(correct_a, dtype=np.float64) - np.asarray(correct_b, dtype=np.float64)
    n = d.size
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = stats.norm.ppf(0.5 + level / 2)
    return {"diff": mean, "se": se, "lo": mean - z * se, "hi": mean + z * se, "n": int(n)}


def tost(correct_a: NDArray, correct_b: NDArray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests that |acc_a - acc_b| < margin.

    Equivalence is declared when both one-sided p-values are below ``alpha``, i.e. when the
    (1 - 2 alpha) interval of the difference lies inside (-margin, margin).
    """
    pd = paired_difference(correct_a, correct_b, level=1 - 2 * alpha)
    if pd["se"] == 0.0:
        p_lower = 0.0 if pd["diff"] > -margin else 1.0
        p_upper = 0.0 if pd["diff"] < margin else 1.0
    else:
        p_lower = float(stats.norm.sf((pd["diff"] + margin) / pd["se"]))
        p_upper = float(stats.norm.cdf((pd["diff"] - margin) / pd["se"]))
    p = max(p_lower, p_upper)
    return {
        **pd,
        "margin": margin,
        "p_value": p,
        "equivalent": bool(p < alpha),
        "ci90": [pd["lo"], pd["hi"]],
    }


def mcnemar_sample_size(
    delta: float, discordance: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Test windows needed for a paired test to detect an accuracy difference ``delta``.

    Normal approximation n = (z_{1-alpha/2} + z_{power})^2 * discordance / delta^2, where
    ``discordance`` is the fraction of windows on which the two classifiers disagree.
    """
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(math.ceil(z**2 * discordance / delta**2))


def logloss_bits(proba_pos: NDArray, y: NDArray, eps: float = 1e-12) -> NDArray[np.float64]:
    """Per-window binary cross-entropy in bits."""
    p = np.clip(np.asarray(proba_pos, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)) / LOG2


def label_entropy_bits(y_train: NDArray) -> float:
    """Entropy of the label marginal (the V-entropy of the constant predictor), in bits."""
    q = float(np.clip(np.mean(y_train), 1e-12, 1 - 1e-12))
    return float(-(q * math.log(q) + (1 - q) * math.log(1 - q)) / LOG2)


def vinfo_bits(proba_pos: NDArray, y_test: NDArray, y_train: NDArray) -> float:
    """Held-out estimate of the predictive V-information I_V(X -> Y), in bits."""
    return label_entropy_bits(y_train) - float(logloss_bits(proba_pos, y_test).mean())


def paired_logloss_difference(
    proba_a: NDArray, proba_b: NDArray, y: NDArray, level: float = 0.95
) -> dict[str, float]:
    """Mean of loss(b) - loss(a) in bits with a paired normal interval.

    Positive values mean probe a is more informative than probe b. With a = (control, Z) and b =
    control this is the conditional V-information I_V(Z -> Y | control).
    """
    d = logloss_bits(proba_b, y) - logloss_bits(proba_a, y)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    z = stats.norm.ppf(0.5 + level / 2)
    return {
        "bits": mean,
        "se": se,
        "lo": mean - z * se,
        "hi": mean + z * se,
        "p_value": float(2 * stats.norm.sf(abs(mean) / se)) if se > 0 else 1.0,
    }
