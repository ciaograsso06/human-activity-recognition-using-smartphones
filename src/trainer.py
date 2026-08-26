from __future__ import annotations

import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .metrics import classification_metrics

LOGGER = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_examples += batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    return (
        total_loss / total_examples,
        total_correct / total_examples,
        y_true,
        y_pred,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: str | Path,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, float]]:
    """Train with validation Macro F1 early stopping and best-checkpoint saving."""
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, patience // 2)
    )

    model = model.to(device)
    best_f1 = -1.0
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    epoch_times: list[float] = []
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc, _, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_true, val_pred = _run_epoch(
            model, val_loader, criterion, device
        )
        val_f1 = float(
            f1_score(val_true, val_pred, labels=list(range(6)), average="macro", zero_division=0)
        )
        scheduler.step(val_f1)
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["learning_rate"].append(learning_rate_now)

        LOGGER.info(
            "Epoch %03d | train_loss=%.4f val_loss=%.4f | train_acc=%.4f "
            "val_acc=%.4f val_macro_f1=%.4f lr=%.2e",
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            val_f1,
            learning_rate_now,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, checkpoint)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    training_time = time.perf_counter() - training_start
    model.load_state_dict(best_state)
    timing = {
        "training_time_seconds": float(training_time),
        "mean_epoch_time_seconds": float(np.mean(epoch_times)),
        "epochs_trained": float(len(epoch_times)),
        "best_validation_macro_f1": float(best_f1),
    }
    return model, history, timing


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate a trained model and include average per-sample inference time."""
    model.eval()
    model.to(device)
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for inputs, targets in loader:
        logits = model(inputs.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()
        y_pred_batches.append(predictions)
        y_true_batches.append(targets.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    metrics = classification_metrics(y_true, y_pred)
    metrics["inference_time_ms_per_sample"] = float(1000.0 * elapsed / len(y_true))
    metrics["parameters"] = count_parameters(model)
    return metrics, y_true, y_pred
