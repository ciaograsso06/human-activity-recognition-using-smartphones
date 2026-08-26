from pathlib import Path

import numpy as np

from src.dataset import SIGNAL_NAMES, _load_split, _normalize


def _write_fake_split(root: Path, split: str, n_samples: int = 5) -> None:
    suffix = split
    inertial = root / split / "Inertial Signals"
    inertial.mkdir(parents=True)
    for index, signal in enumerate(SIGNAL_NAMES):
        values = np.full((n_samples, 128), index + 1, dtype=np.float32)
        np.savetxt(inertial / f"{signal}_{suffix}.txt", values)
    np.savetxt(root / split / f"y_{suffix}.txt", np.arange(n_samples) % 6 + 1, fmt="%d")
    np.savetxt(root / split / f"subject_{suffix}.txt", np.arange(n_samples) + 1, fmt="%d")


def test_dataset_loading_shape(tmp_path: Path) -> None:
    _write_fake_split(tmp_path, "train")
    x, y, subjects = _load_split(tmp_path, "train")
    assert x.shape == (5, 128, 9)
    assert y.shape == (5,)
    assert subjects.shape == (5,)
    assert y.min() == 0


def test_normalization_uses_train_statistics() -> None:
    train = np.arange(4 * 128 * 9, dtype=np.float32).reshape(4, 128, 9)
    val = train[:1] + 10
    test = train[:1] - 10
    norm_train, norm_val, norm_test, mean, std = _normalize(train, val, test)
    assert norm_train.shape == train.shape
    assert norm_val.shape == val.shape
    assert norm_test.shape == test.shape
    assert mean.shape == (9,)
    assert std.shape == (9,)
    assert np.allclose(norm_train.mean(axis=(0, 1)), 0.0, atol=1e-5)
