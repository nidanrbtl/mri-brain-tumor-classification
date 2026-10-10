"""Eğitim ve değerlendirme grafikleri. TensorFlow gerektirmez."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ekrana değil dosyaya çiz

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from src.utils import ensure_dirs, project_path

# Seri renkleri: renk körlüğünde de ayırt edilebilen sabit bir sıra
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _save(output_path: Path) -> Path:
    ensure_dirs(output_path.parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_training_history(history: pd.DataFrame, model_name: str, title: str) -> tuple[Path, Path]:
    """Eğitim ve doğrulama doğruluğu / kaybı eğrilerini iki ayrı dosyaya kaydeder."""
    figure_dir = project_path(config.FIGURE_DIR)
    epochs = np.arange(1, len(history) + 1)
    paths = []

    for metric, label in (("accuracy", "Doğruluk"), ("loss", "Kayıp")):
        plt.figure(figsize=(9, 6))
        plt.plot(epochs, history[metric], label="Eğitim", color=SERIES_COLORS[0], linewidth=2, marker="o", markersize=3)
        plt.plot(epochs, history[f"val_{metric}"], label="Doğrulama", color=SERIES_COLORS[1], linewidth=2,
                 marker="o", markersize=3)
        plt.title(f"{title} - {label}")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        paths.append(_save(figure_dir / f"{model_name}_{metric}.png"))

    return paths[0], paths[1]


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, output_path: Path) -> Path:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    return _save(output_path)


def plot_model_comparison(rows: list[dict], output_path: Path) -> Path:
    """
    Modellerin doğrulama ve test metriklerini yan yana gösterir.
    Her satır: display_name, val_accuracy, test_accuracy, macro_f1, binary_accuracy.
    """
    metrics = [
        ("val_accuracy", "Doğrulama doğruluğu"),
        ("test_accuracy", "Test doğruluğu"),
        ("macro_f1", "Makro F1 (test)"),
        ("binary_accuracy", "Tümör var/yok doğruluğu (test)"),
    ]
    x = np.arange(len(rows))
    width = 0.17

    plt.figure(figsize=(10, 6))
    for i, (key, label) in enumerate(metrics):
        values = [row[key] for row in rows]
        bars = plt.bar(x + (i - 1.5) * (width + 0.02), values, width, label=label, color=SERIES_COLORS[i])
        plt.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=9, padding=2)

    plt.xticks(x, [row["display_name"] for row in rows])
    plt.ylim(0, 1.08)
    plt.ylabel("Skor")
    plt.title("Model karşılaştırması")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    plt.grid(True, axis="y", alpha=0.3)
    return _save(output_path)


def plot_misclassified_examples(
    filepaths: list[str], true_labels: list[str], predicted_labels: list[str],
    model_key: str, title: str, max_examples: int = 9,
) -> Path | None:
    """Yanlış sınıflandırılan ilk örnekleri bir ızgarada gösterir; hata yoksa None döner."""
    wrong = [i for i, (t, p) in enumerate(zip(true_labels, predicted_labels)) if t != p][:max_examples]
    if not wrong:
        return None

    cols = 3
    rows = int(np.ceil(len(wrong) / cols))
    _, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, index in zip(axes, wrong):
        image = plt.imread(project_path(filepaths[index]))
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(f"Gerçek: {true_labels[index]}\nTahmin: {predicted_labels[index]}", fontsize=9)
    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"{title} - yanlış sınıflandırılan örnekler", fontsize=12)
    return _save(project_path(config.FIGURE_DIR) / f"{model_key}_misclassified_examples.png")


def plot_class_distribution(counts: dict[str, int], output_path: Path) -> Path:
    plt.figure(figsize=(9, 6))
    bars = plt.bar(list(counts), list(counts.values()), width=0.5, color=SERIES_COLORS[0])
    plt.bar_label(bars, padding=3)
    plt.title("Sınıf başına görüntü sayısı")
    plt.ylabel("Görüntü sayısı")
    plt.grid(True, axis="y", alpha=0.3)
    return _save(output_path)


def plot_sample_images(records: pd.DataFrame, output_path: Path, per_class: int = 3) -> Path:
    """Her sınıftan birkaç örnek görüntüyü ızgarada gösterir."""
    _, axes = plt.subplots(len(config.CLASS_NAMES), per_class, figsize=(3 * per_class, 3 * len(config.CLASS_NAMES)))
    axes = np.array(axes).reshape(len(config.CLASS_NAMES), per_class)

    for row, class_name in enumerate(config.CLASS_NAMES):
        paths = records.loc[records["label"] == class_name, "filepath"].head(per_class).tolist()
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(paths):
                image = plt.imread(project_path(paths[col]))
                ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
                if col == 0:
                    ax.set_title(class_name, fontsize=11, loc="left")

    plt.suptitle("Sınıflardan örnek MRI görüntüleri", fontsize=14)
    return _save(output_path)
