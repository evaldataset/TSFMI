"""Probing evaluation metrics for classification, regression, and representation analysis."""

from __future__ import annotations

from typing import cast

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compute classification probe metrics.

    Args:
        logits: Raw model outputs of shape (N, num_classes) or (N,) for binary.
        labels: Integer ground-truth labels of shape (N,).

    Returns:
        Dict with keys: "accuracy", "f1_macro", "f1_weighted".
    """
    # All training paths in this repo emit logits of shape (N, num_classes) and
    # use CrossEntropyLoss, so the multi-class branch below is the live path.
    # The shape[-1]==1 / 1-D fallback is retained for external callers that
    # supply a single-logit binary head.
    if logits.ndim == 1 or logits.shape[-1] == 1:
        preds = (logits.flatten() > 0).long()
    else:
        preds = torch.argmax(logits, dim=-1)

    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    return {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "f1_macro": float(f1_score(labels_np, preds_np, average="macro")),
        "f1_weighted": float(f1_score(labels_np, preds_np, average="weighted")),
    }


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Compute regression probe metrics.

    Args:
        predictions: Predicted values of shape (N,) or (N, 1).
        targets: Ground-truth float values of shape (N,).

    Returns:
        Dict with keys: "r2", "mae", "mse".
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    return {
        "r2": float(r2_score(targets_np, preds_np)),
        "mae": float(mean_absolute_error(targets_np, preds_np)),
        "mse": float(mean_squared_error(targets_np, preds_np)),
    }


def compute_selectivity(probe_acc: float, control_acc: float) -> float:
    """Compute probe selectivity (Hewitt & Liang, 2019).

    Selectivity = linear_probe_accuracy - control_probe_accuracy.
    Measures how much accuracy is due to LINEAR accessibility of the property,
    as opposed to memorization or nonlinear pattern recognition.

    Args:
        probe_acc: Accuracy of the linear probe.
        control_acc: Accuracy of the MLP control probe.

    Returns:
        Selectivity score (can be negative if control outperforms linear probe).
    """
    return probe_acc - control_acc


def _hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute unbiased HSIC estimator with linear kernel.

    Args:
        x: Centered matrix of shape (N, d1).
        y: Centered matrix of shape (N, d2).

    Returns:
        Scalar HSIC value.
    """
    n = x.shape[0]
    cross = x.T @ y  # (d1, d2)
    # cast: Tensor/int division return type is Any in torch stubs; runtime always returns Tensor
    return cast(torch.Tensor, (cross.norm() ** 2) / ((n - 1) ** 2))


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear Centered Kernel Alignment (CKA) between two representation matrices.

    Measures representational similarity between two sets of features. CKA=1 means
    identical representations (up to orthogonal transformation), CKA=0 means no similarity.

    Uses the linear kernel: K = X @ X^T.

    Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations
    Revisited." ICML 2019.

    Args:
        X: First representation matrix of shape (N, d1).
        Y: Second representation matrix of shape (N, d2).

    Returns:
        CKA similarity score in [0, 1].

    Raises:
        ValueError: If X and Y have different number of samples (N must match).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, got {X.shape[0]} and {Y.shape[0]}"
        )

    x = X.detach().float()
    y = Y.detach().float()

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    hsic_xy = _hsic(x, y)
    hsic_xx = _hsic(x, x)
    hsic_yy = _hsic(y, y)

    denominator = torch.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-10:
        return 0.0

    cka = hsic_xy / denominator
    return float(cka.clamp(0.0, 1.0))


def compute_cka_with_ci(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute CKA with bootstrap confidence interval.

    Args:
        X: First representation matrix (N, d1).
        Y: Second representation matrix (N, d2).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (cka_point, ci_low, ci_high).
    """
    import numpy as np

    n = X.shape[0]
    rng = np.random.default_rng(seed)

    bootstrap_ckas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        cka_b = compute_cka(X[idx], Y[idx])
        bootstrap_ckas.append(cka_b)

    alpha = 1.0 - ci
    # Use bootstrap median as point estimate to guarantee ci_low <= cka <= ci_high
    cka_point = float(np.median(bootstrap_ckas))
    ci_low = float(np.percentile(bootstrap_ckas, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_ckas, 100 * (1 - alpha / 2)))
    return cka_point, ci_low, ci_high
