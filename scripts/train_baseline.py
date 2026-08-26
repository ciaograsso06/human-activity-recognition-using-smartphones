from __future__ import annotations

import _bootstrap

import logging
from pathlib import Path

from src.dataset import create_dataloaders
from src.models import CNNBaseline
from src.trainer import evaluate_model, train_model
from src.utils import get_device, load_config, save_json, set_seed
from src.visualization import plot_confusion_matrix, plot_training_curves


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    data = create_dataloaders(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["training"]["batch_size"],
        validation_size=cfg["data"]["validation_size"],
        normalize=cfg["data"]["normalize"],
        seed=cfg["seed"],
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"Train subjects: {data.train_subjects.tolist()}")
    print(f"Validation subjects: {data.val_subjects.tolist()}")

    model = CNNBaseline(
        cnn_channels=cfg["model"]["cnn_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["classifier_dropout"],
    )
    model, history, timing = train_model(
        model,
        data.train_loader,
        data.val_loader,
        device,
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
        checkpoint_path="checkpoints/cnn_baseline.pt",
    )
    metrics, y_true, y_pred = evaluate_model(model, data.test_loader, device)
    metrics.update(timing)
    metrics["model"] = "CNN"
    metrics["heads"] = None
    metrics["embed_dim"] = None

    save_json(metrics, "results/baseline_metrics.json")
    plot_training_curves(history, "results/baseline_training_curves.png")
    plot_confusion_matrix(y_true, y_pred, "results/baseline_confusion_matrix.png", "CNN baseline")
    print(f"Test Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
