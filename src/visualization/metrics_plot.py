"""Figures for multi-label training and held-out evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.data.ptbxl import SUPERCLASSES


def plot_class_distribution(labels: np.ndarray, output_path: str | Path) -> None:
    counts = np.asarray(labels).sum(axis=0)
    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(SUPERCLASSES, counts, color="#4263eb")
    axis.bar_label(bars, fmt="%.0f")
    axis.set_ylabel("Training records with label")
    axis.set_title("PTB-XL diagnostic superclass distribution (train folds only)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_training_curves(history: list[dict], output_path: str | Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], marker="o", label="Train")
    axes[0].plot(epochs, [entry["validation_loss"] for entry in history], marker="o", label="Validation")
    axes[0].set_title("BCE loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [entry["validation_macro_auroc"] for entry in history], marker="o")
    axes[1].set_title("Validation macro AUROC")
    axes[1].set_xlabel("Epoch")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_roc_curves(targets: np.ndarray, probabilities: np.ndarray, output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(SUPERCLASSES):
        if np.unique(targets[:, index]).size != 2:
            continue
        fpr, tpr, _ = roc_curve(targets[:, index], probabilities[:, index])
        axis.plot(fpr, tpr, label=f"{name} (AUROC {auc(fpr, tpr):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curves")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrices(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    output_path: str | Path,
) -> None:
    figure, axes = plt.subplots(1, len(SUPERCLASSES), figsize=(17, 3.5))
    for index, (axis, name) in enumerate(zip(axes, SUPERCLASSES)):
        predicted = probabilities[:, index] >= thresholds[index]
        matrix = confusion_matrix(targets[:, index], predicted, labels=[0, 1])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_xticks([0, 1], labels=["0", "1"])
        axis.set_yticks([0, 1], labels=["0", "1"])
        axis.set_title(name)
        axis.set_xlabel("Predicted")
    axes[0].set_ylabel("True label")
    figure.suptitle("Test confusion matrices (one binary task per superclass)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
