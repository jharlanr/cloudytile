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
) -> tuple[float, dict]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        loader: Data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)

            total_loss += loss.item() * images.size(0)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return avg_loss, metrics
