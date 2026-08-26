from __future__ import annotations

import _bootstrap

import logging

from src.dataset import create_dataloaders
from src.models import CNNMultiHeadAttention
from src.optimization import optimize_attention, plot_optimization_history
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
    study = optimize_attention(
        data=data,
        device=device,
        n_trials=cfg["optimization"]["n_trials"],
        epochs_per_trial=cfg["optimization"]["epochs_per_trial"],
        patience=cfg["optimization"]["patience"],
        weight_decay=cfg["training"]["weight_decay"],
        output_dir="checkpoints/optuna_trials",
        seed=cfg["seed"],
    )
    best_params = dict(study.best_params)
    save_json(
        {
            "best_validation_macro_f1": float(study.best_value),
            "best_params": best_params,
        },
        "results/best_params.json",
    )
    plot_optimization_history(study, "results/optuna_history.png")

    model = CNNMultiHeadAttention(
        cnn_channels=best_params["cnn_channels"],
        kernel_size=best_params["kernel_size"],
        embed_dim=best_params["embed_dim"],
        num_heads=best_params["num_heads"],
        num_attention_layers=best_params["num_attention_layers"],
        attention_dropout=best_params["attention_dropout"],
        classifier_dropout=best_params["classifier_dropout"],
        feedforward_dim=best_params["feedforward_dim"],
    )
    model, history, timing = train_model(
        model,
        data.train_loader,
        data.val_loader,
        device,
        epochs=cfg["training"]["epochs"],
        learning_rate=best_params["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
        checkpoint_path="checkpoints/cnn_mha_optimized.pt",
    )
    metrics, y_true, y_pred = evaluate_model(model, data.test_loader, device)
    metrics.update(timing)
    metrics["model"] = "Optimized MHA"
    metrics["heads"] = best_params["num_heads"]
    metrics["embed_dim"] = best_params["embed_dim"]
    metrics["best_params"] = best_params
    save_json(metrics, "results/optimized_metrics.json")
    plot_training_curves(history, "results/optimized_training_curves.png")
    plot_confusion_matrix(
        y_true, y_pred, "results/optimized_confusion_matrix.png", "Optimized CNN + MHA"
    )
    print("Best parameters:", best_params)
    print(f"Test Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
