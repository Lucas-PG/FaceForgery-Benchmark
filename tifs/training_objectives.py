"""Matched loss functions on B recordings x K video-level signed logits."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

OBJECTIVES = {
    "clean",
    "augmented",
    "worst_view",
    "partial_auc",
    "constrained",
    "shuffled",
    "per_channel",
    "no_clean",
    "frozen",
}
CONSTRAINED = {"constrained", "shuffled", "per_channel", "no_clean"}


def loss_for_batch(
    logits, labels, objective: str, settings: dict, dual: torch.Tensor, floor: float
):
    """Return loss and detached *signed* violations for projected dual ascent."""
    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("Expected one row of route logits per recording")
    positive, negative = logits[labels == 1], logits[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("Every training batch must contain both classes")
    targets = labels.float()[:, None].expand_as(logits)
    average = F.binary_cross_entropy_with_logits(logits, targets)
    if objective in {"clean", "augmented"}:
        return average, None
    if objective == "worst_view":
        return F.binary_cross_entropy_with_logits(
            torch.where(labels.bool(), logits.amin(1), logits.amax(1)), labels.float()
        ), None
    alpha = float(settings.get("alpha", 0.05))
    if objective == "partial_auc":
        # Pairwise logistic ranking surrogate restricted to top-alpha genuine tail.
        # This baseline treats route exposures marginally, with identical forward pass.
        all_negative = negative.flatten()
        top = all_negative.topk(max(1, math.ceil(alpha * all_negative.numel()))).values
        return F.softplus(top[None, :] - positive.flatten()[:, None]).mean(), None
    if objective not in CONSTRAINED:
        raise ValueError(f"Unknown learning objective {objective}")
    temperature = float(settings.get("temperature", 0.3))
    catch = torch.sigmoid(positive.amin(1) / temperature).mean()
    if objective == "per_channel":
        false_alert = torch.sigmoid(negative / temperature).mean(0)
    else:
        false_alert = torch.sigmoid(negative.amax(1) / temperature).mean().reshape(1)
    constraints = false_alert - alpha
    if objective != "no_clean":
        clean_catch = torch.sigmoid(positive[:, 0] / temperature).mean()
        constraints = torch.cat((constraints, (floor - clean_catch).reshape(1)))
    # The BCE anchor prevents saturated rate surrogates from erasing the ranking
    # gradient; the operating-point intercept is learned via the detector's head.
    loss = (
        1
        - catch
        + torch.dot(dual, constraints)
        + float(settings.get("bce_anchor", 0.2)) * average
    )
    return loss, constraints.detach()
