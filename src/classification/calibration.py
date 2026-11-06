"""
Probability Calibration and Temperature Scaling for Fine-Grained Vehicle Classification.
Implements post-hoc temperature scaling to calibrate confidence estimates on the validation set,
computing Expected Calibration Error (ECE) and Negative Log Likelihood (NLL).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class ModelWithTemperature(nn.Module):
    """
    Decorator module that wraps a trained classifier with a learned temperature scalar T.
    Calibrates softmax probabilities without altering classification accuracy (order-preserving).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by 1/T."""
        # Expand temperature to match logits shape
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def calibrate(
        self,
        val_loader: Any,
        device: torch.device | str = "cpu",
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> dict[str, float]:
        """
        Tune temperature T on validation set using L-BFGS optimizer minimizing NLL.

        Returns:
            Dictionary with calibration statistics: initial_temp, optimal_temp,
            nll_before, nll_after, ece_before, ece_after.
        """
        self.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)

        # 1. Collect all logits and labels from validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(device)
                    labels = batch["class_id"].to(device)
                else:
                    images, labels = batch[0].to(device), batch[1].to(device)

                # Filter invalid labels (-1)
                valid_mask = labels >= 0
                if not valid_mask.any():
                    continue

                logits = self.model(images[valid_mask])
                logits_list.append(logits)
                labels_list.append(labels[valid_mask])

        if not logits_list:
            logger.warning("No valid samples found for temperature calibration.")
            return {
                "optimal_temperature": 1.0,
                "nll_before": 0.0,
                "nll_after": 0.0,
                "ece_before": 0.0,
                "ece_after": 0.0,
            }

        all_logits = torch.cat(logits_list).to(device)
        all_labels = torch.cat(labels_list).to(device)

        # Calculate metrics before scaling
        nll_before = float(nll_criterion(all_logits, all_labels).item())
        probs_before = torch.softmax(all_logits, dim=1).cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        ece_before = compute_ece(probs_before, labels_np)

        # 2. Optimize temperature with L-BFGS
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Clamp temperature to reasonable positive bounds
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)

        # Calculate metrics after scaling
        scaled_logits = self.temperature_scale(all_logits)
        nll_after = float(nll_criterion(scaled_logits, all_labels).item())
        probs_after = torch.softmax(scaled_logits, dim=1).detach().cpu().numpy()
        ece_after = compute_ece(probs_after, labels_np)

        opt_temp = float(self.temperature.item())
        logger.info(
            "Calibration complete. T=%.4f | NLL: %.4f -> %.4f | ECE: %.4f -> %.4f",
            opt_temp,
            nll_before,
            nll_after,
            ece_before,
            ece_after,
        )

        return {
            "optimal_temperature": round(opt_temp, 4),
            "nll_before": round(nll_before, 4),
            "nll_after": round(nll_after, 4),
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
        }


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) across confidence bins.

    Args:
        probs: Softmax probabilities of shape (N, C).
        labels: Ground truth class integers of shape (N,).
        n_bins: Number of equally spaced confidence bins in [0, 1].

    Returns:
        Weighted scalar ECE value.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
