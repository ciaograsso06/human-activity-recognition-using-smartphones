from __future__ import annotations

import _bootstrap

import logging

from src.dataset import create_dataloaders
from src.models import CNNMultiHeadAttention
from src.trainer import evaluate_model, train_model
from src.utils import get_device, load_config, save_json, set_seed
from src.visualization import (
    plot_attention_distribution,
    plot_confusion_matrix,
    plot_training_curves,
)


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
    model = CNNMultiHeadAttention(
        cnn_channels=cfg["model"]["cnn_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_attention_layers=cfg["model"]["num_attention_layers"],
        attention_dropout=cfg["model"]["attention_dropout"],
        classifier_dropout=cfg["model"]["classifier_dropout"],
        feedforward_dim=cfg["model"]["feedforward_dim"],
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
        checkpoint_path="checkpoints/cnn_mha_fixed.pt",
    )
    metrics, y_true, y_pred = evaluate_model(model, data.test_loader, device)
    metrics.update(timing)
    metrics["model"] = "CNN + MHA"
    metrics["heads"] = cfg["model"]["num_heads"]
    metrics["embed_dim"] = cfg["model"]["embed_dim"]

    save_json(metrics, "results/attention_metrics.json")
    plot_training_curves(history, "results/attention_training_curves.png")
    plot_confusion_matrix(y_true, y_pred, "results/attention_confusion_matrix.png", "CNN + MHA")

    inputs, _ = next(iter(data.test_loader))
    attention = model.attention_weights(inputs[:16].to(device))[0].cpu().numpy()
    plot_attention_distribution(attention, "results/attention_distribution.png")
    print(f"Test Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
