from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

SIGNAL_NAMES = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)

ACTIVITY_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)


class HARDataset(Dataset):
    """PyTorch dataset for UCI HAR inertial windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    mean: np.ndarray
    std: np.ndarray
    train_subjects: np.ndarray
    val_subjects: np.ndarray


def _load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inertial_dir = root / split / "Inertial Signals"
    if not inertial_dir.exists():
        raise FileNotFoundError(
            f"Missing '{inertial_dir}'. Run: python scripts/download_data.py"
        )

    suffix = "train" if split == "train" else "test"
    channels = [
        np.loadtxt(inertial_dir / f"{name}_{suffix}.txt", dtype=np.float32)
        for name in SIGNAL_NAMES
    ]
    x = np.stack(channels, axis=-1)  # [N, 128, 9]
    y = np.loadtxt(root / split / f"y_{suffix}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{suffix}.txt", dtype=np.int64)
    return x, y, subjects


def _split_train_by_subject(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(x, y, groups=subjects))
    return (
        x[train_idx],
        y[train_idx],
        subjects[train_idx],
        x[val_idx],
        y[val_idx],
        subjects[val_idx],
    )


def _normalize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (x_train - mean) / std,
        (x_val - mean) / std,
        (x_test - mean) / std,
        mean.squeeze((0, 1)),
        std.squeeze((0, 1)),
    )


def create_dataloaders(
    root_dir: str | Path,
    batch_size: int = 64,
    validation_size: float = 0.15,
    normalize: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> DataBundle:
    """Load UCI HAR and build subject-disjoint train/validation loaders."""
    root = Path(root_dir)
    x_train_all, y_train_all, subjects = _load_split(root, "train")
    x_test, y_test, _ = _load_split(root, "test")

    x_train, y_train, train_subjects, x_val, y_val, val_subjects = (
        _split_train_by_subject(
            x_train_all,
            y_train_all,
            subjects,
            validation_size=validation_size,
            seed=seed,
        )
    )

    if normalize:
        x_train, x_val, x_test, mean, std = _normalize(x_train, x_val, x_test)
    else:
        mean = np.zeros(len(SIGNAL_NAMES), dtype=np.float32)
        std = np.ones(len(SIGNAL_NAMES), dtype=np.float32)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        HARDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        HARDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        HARDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        mean=mean,
        std=std,
        train_subjects=np.unique(train_subjects),
        val_subjects=np.unique(val_subjects),
    )
