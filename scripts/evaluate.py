from __future__ import annotations

import _bootstrap

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.dataset import create_dataloaders
from src.models import CNNBaseline, CNNMultiHeadAttention
from src.trainer import evaluate_model, train_model
from src.utils import get_device, load_config, save_json, set_seed
from src.visualization import plot_heads_comparison, plot_model_f1


def _read_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": metrics["model"],
        "heads": metrics.get("heads"),
        "embed_dim": metrics.get("embed_dim"),
        "parameters": metrics["parameters"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "training_time_seconds": metrics.get("training_time_seconds"),
        "mean_epoch_time_seconds": metrics.get("mean_epoch_time_seconds"),
        "inference_time_ms_per_sample": metrics["inference_time_ms_per_sample"],
    }


def _train_head_ablation(
    heads: int,
    cfg: dict[str, Any],
    data: Any,
    device: torch.device,
) -> dict[str, Any]:
    model = CNNMultiHeadAttention(
        cnn_channels=cfg["model"]["cnn_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=heads,
        num_attention_layers=cfg["model"]["num_attention_layers"],
        attention_dropout=cfg["model"]["attention_dropout"],
        classifier_dropout=cfg["model"]["classifier_dropout"],
        feedforward_dim=cfg["model"]["feedforward_dim"],
    )
    model, _, timing = train_model(
        model,
        data.train_loader,
        data.val_loader,
        device,
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
        checkpoint_path=f"checkpoints/ablation_heads_{heads}.pt",
    )
    metrics, _, _ = evaluate_model(model, data.test_loader, device)
    metrics.update(timing)
    metrics["model"] = f"CNN + MHA ({heads} head{'s' if heads != 1 else ''})"
    metrics["heads"] = heads
    metrics["embed_dim"] = cfg["model"]["embed_dim"]
    save_json(metrics, f"results/ablation_heads_{heads}.json")
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    data = create_dataloaders(
        root_dir=cfg["data"]["root_dir"],
        batch_size=cfg["training"]["batch_size"],
        validation_size=cfg["data"]["validation_size"],
        normalize=cfg["data"]["normalize"],
        seed=cfg["seed"],
        num_workers=cfg["data"]["num_workers"],
    )

    required = [
        "results/baseline_metrics.json",
        "results/attention_metrics.json",
        "results/optimized_metrics.json",
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing previous experiment results: "
            + ", ".join(missing)
            + ". Run train_baseline.py, train_attention.py and optimize.py first."
        )

    baseline = _read_json("results/baseline_metrics.json")
    fixed_attention = _read_json("results/attention_metrics.json")
    optimized = _read_json("results/optimized_metrics.json")

    ablations: list[dict[str, Any]] = []
    for heads in [1, 2, 4, 8]:
        if heads == cfg["model"]["num_heads"]:
            metric = dict(fixed_attention)
            metric["model"] = f"CNN + MHA ({heads} heads)"
            ablations.append(metric)
            continue
        saved_path = Path(f"results/ablation_heads_{heads}.json")
        if saved_path.exists():
            ablations.append(_read_json(str(saved_path)))
        else:
            ablations.append(_train_head_ablation(heads, cfg, data, device))

    all_metrics = [baseline, *ablations, optimized]
    frame = pd.DataFrame([_summary_row(metric) for metric in all_metrics])
    frame.to_csv("results/experiments.csv", index=False)
    plot_heads_comparison(frame, "results/heads_comparison.png")
    plot_model_f1(frame, "results/f1_by_architecture.png")

    final_metrics = {
        "train_subjects": data.train_subjects.tolist(),
        "validation_subjects": data.val_subjects.tolist(),
        "experiments": frame.to_dict(orient="records"),
    }
    save_json(final_metrics, "results/metrics.json")
    print(frame.to_string(index=False))
    print("\nSaved: results/experiments.csv and results/metrics.json")


if __name__ == "__main__":
    main()
