from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import optuna
import torch

from .dataset import DataBundle
from .models import CNNMultiHeadAttention
from .trainer import train_model


def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
    compatible_embed_dims = [dim for dim in [32, 64, 128, 256] if dim % num_heads == 0]
    embed_dim = trial.suggest_categorical("embed_dim", compatible_embed_dims)
    return {
        "cnn_channels": trial.suggest_categorical("cnn_channels", [32, 64, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_attention_layers": trial.suggest_int("num_attention_layers", 1, 3),
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.4),
        "classifier_dropout": trial.suggest_float("classifier_dropout", 0.0, 0.5),
        "feedforward_dim": trial.suggest_categorical("feedforward_dim", [128, 256, 512]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
    }


def optimize_attention(
    data: DataBundle,
    device: torch.device,
    n_trials: int,
    epochs_per_trial: int,
    patience: int,
    weight_decay: float,
    output_dir: str | Path,
    seed: int,
) -> optuna.Study:
    """Optimize CNN+MHA architecture using validation Macro F1."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        model = CNNMultiHeadAttention(
            cnn_channels=params["cnn_channels"],
            kernel_size=params["kernel_size"],
            embed_dim=params["embed_dim"],
            num_heads=params["num_heads"],
            num_attention_layers=params["num_attention_layers"],
            attention_dropout=params["attention_dropout"],
            classifier_dropout=params["classifier_dropout"],
            feedforward_dim=params["feedforward_dim"],
        )
        _, history, _ = train_model(
            model=model,
            train_loader=data.train_loader,
            val_loader=data.val_loader,
            device=device,
            epochs=epochs_per_trial,
            learning_rate=params["learning_rate"],
            weight_decay=weight_decay,
            patience=patience,
            checkpoint_path=output / f"trial_{trial.number}.pt",
        )
        for step, value in enumerate(history["val_macro_f1"]):
            trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return max(history["val_macro_f1"])

    study.optimize(objective, n_trials=n_trials)
    return study


def plot_optimization_history(study: optuna.Study, output_path: str | Path) -> None:
    """Plot completed-trial values without extra visualization dependencies."""
    completed = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        return
    trial_numbers = [trial.number for trial in completed]
    values = [float(trial.value) for trial in completed if trial.value is not None]
    best_so_far: list[float] = []
    running = float("-inf")
    for value in values:
        running = max(running, value)
        best_so_far.append(running)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trial_numbers, values, marker="o", label="Trial Macro F1")
    ax.plot(trial_numbers, best_so_far, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Optuna optimization history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
