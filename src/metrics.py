from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .dataset import ACTIVITY_NAMES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute the metrics used across all experiments."""
    labels = list(range(len(ACTIVITY_NAMES)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    def directed_error(source: int, target: int) -> dict[str, float | int]:
        source_total = int(cm[source].sum())
        count = int(cm[source, target])
        return {
            "count": count,
            "rate": float(count / source_total) if source_total else 0.0,
        }

    key_confusions = {
        "SITTING_as_STANDING": directed_error(3, 4),
        "STANDING_as_SITTING": directed_error(4, 3),
        "WALKING_as_WALKING_UPSTAIRS": directed_error(0, 1),
        "WALKING_as_WALKING_DOWNSTAIRS": directed_error(0, 2),
        "WALKING_UPSTAIRS_as_WALKING": directed_error(1, 0),
        "WALKING_UPSTAIRS_as_WALKING_DOWNSTAIRS": directed_error(1, 2),
        "WALKING_DOWNSTAIRS_as_WALKING": directed_error(2, 0),
        "WALKING_DOWNSTAIRS_as_WALKING_UPSTAIRS": directed_error(2, 1),
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "key_confusions": key_confusions,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=ACTIVITY_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
