"""
Training utilities for CloudyTileCNN.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_metrics(
    all_labels: list,
    all_preds: list,
    all_probs: list,
) -> dict:
    """
    Compute classification metrics from predictions.

    Args:
        all_labels: Ground truth labels (0 or 1)
        all_preds: Predicted labels (0 or 1)
        all_probs: Predicted probabilities for class 1

    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    # AUC requires both classes to be present
    if len(set(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc"] = 0.0

    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Args:
        model: The model to train
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        threshold: Probability above which a tile is called useful. 0.5 is only
            the right operating point if false positives and false negatives
            cost the same; they do not here (see pick_threshold).

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    avg_loss, all_labels, all_probs = predict_probs(model, loader, criterion, device)
    all_preds = [float(p >= threshold) for p in all_probs]
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["threshold"] = threshold

    return avg_loss, metrics


def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list, list]:
    """
    Run the model over a loader once.

    Returns:
        (average_loss, labels, probabilities) — probabilities are kept so an
        operating point can be chosen after the fact rather than baked in.
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return total_loss / len(loader.dataset), all_labels, all_probs


def pick_threshold(
    labels: list,
    probs: list,
    objective: str = "f1",
    target_precision: float = 0.95,
) -> tuple[float, dict]:
    """
    Choose a decision threshold on held-out probabilities.

    The classes here are only ~2.2:1, which is mild — the reason to move off
    0.5 is not imbalance but asymmetric cost. A false positive lets a cloudy
    frame into lake-vision's timeseries and corrupts a drainage call; a false
    negative merely drops one observation from a lake that has ~90 usable ones.
    So precision on the useful class is worth more than recall.

    Tuning the threshold on validation beats weighting the loss: one training
    run yields the whole precision/recall curve, the operating point stays a
    single reportable number, and it can be changed later without retraining.

    Args:
        objective: "f1" maximizes F1; "target_precision" picks the lowest
            threshold whose precision >= target_precision (maximizing recall
            subject to a precision floor), falling back to max-precision if
            the target is unreachable.

    Returns:
        (threshold, metrics_at_that_threshold)
    """
    import numpy as np

    y = np.asarray(labels)
    p = np.asarray(probs)
    candidates = np.unique(np.round(p, 4))
    if len(candidates) > 1:
        candidates = np.concatenate([candidates, [candidates.max() + 1e-6]])

    best_t, best_key, best_metrics = 0.5, -1.0, None
    for t in candidates:
        preds = (p >= t).astype(float)
        m = compute_metrics(y.tolist(), preds.tolist(), p.tolist())
        if objective == "f1":
            key = m["f1"]
        elif objective == "target_precision":
            # rank by recall among thresholds that clear the precision floor;
            # if none do, fall back to whichever gets closest on precision
            key = (1.0 + m["recall"]) if m["precision"] >= target_precision \
                else m["precision"] - 1.0
        else:
            raise ValueError(f"unknown objective {objective!r}")
        if key > best_key:
            best_t, best_key, best_metrics = float(t), key, m

    best_metrics["threshold"] = best_t
    return best_t, best_metrics
