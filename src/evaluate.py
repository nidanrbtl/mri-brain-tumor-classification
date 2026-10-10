"""
Test seti değerlendirmesi: dört sınıflı metrikler, bunlardan türetilen
tümör var/yok metrikleri, karışıklık matrisleri ve model karşılaştırması.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
from src import plots
from src.utils import ensure_dirs, project_path

BINARY_LABELS = ["no_tumor", "tumor"]


def to_binary(class_indices: np.ndarray) -> np.ndarray:
    """Dört sınıflı indeksleri tümör yok (0) / tümör var (1) etiketine çevirir."""
    no_tumor_index = config.CLASS_NAMES.index(config.NO_TUMOR_CLASS)
    return (np.asarray(class_indices) != no_tumor_index).astype(int)


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(config.CLASS_NAMES)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "report_text": classification_report(
            y_true, y_pred, labels=labels, target_names=config.CLASS_NAMES, zero_division=0
        ),
    }


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tümör var/yok metrikleri. Pozitif sınıf 'tümör var'dır."""
    true_binary, pred_binary = to_binary(y_true), to_binary(y_pred)
    cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(true_binary, pred_binary),
        "precision": precision_score(true_binary, pred_binary, zero_division=0),
        "recall": recall_score(true_binary, pred_binary, zero_division=0),  # yakalanan tümör oranı
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),       # doğru tanınan sağlıklı oranı
        "f1": f1_score(true_binary, pred_binary, zero_division=0),
        "confusion_matrix": cm,
        "report_text": classification_report(
            true_binary, pred_binary, labels=[0, 1], target_names=BINARY_LABELS, zero_division=0
        ),
    }


def most_confused_pairs(cm: np.ndarray, top: int = 5) -> list[str]:
    """En sık karıştırılan (gerçek -> tahmin) sınıf çiftleri."""
    pairs = [
        (int(cm[i, j]), config.CLASS_NAMES[i], config.CLASS_NAMES[j])
        for i in range(len(cm)) for j in range(len(cm))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(reverse=True)
    return [f"{true} -> {predicted} ({count})" for count, true, predicted in pairs[:top]]


def best_validation_accuracy(model_key: str) -> float | None:
    """Modelin eğitim geçmişindeki en yüksek doğrulama doğruluğu (kayıt yoksa None)."""
    from src.train import read_history

    history = read_history(model_key)
    return None if history.empty else float(history["val_accuracy"].max())


def report_predictions(model_key: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, dict]:
    """Tahminlerden metrikleri hesaplar; karışıklık matrisi grafiklerini ve metin raporlarını kaydeder."""
    multi = compute_multiclass_metrics(y_true, y_pred)
    binary = compute_binary_metrics(y_true, y_pred)

    figure_dir, report_dir = project_path(config.FIGURE_DIR), project_path(config.REPORT_DIR)
    ensure_dirs(figure_dir, report_dir)
    name = config.FINAL_MODELS[model_key]["display_name"]

    plots.plot_confusion_matrix(
        multi["confusion_matrix"], config.CLASS_NAMES, f"{name} - karışıklık matrisi",
        figure_dir / f"{model_key}_confusion_matrix.png",
    )
    plots.plot_confusion_matrix(
        binary["confusion_matrix"], BINARY_LABELS, f"{name} - tümör var/yok",
        figure_dir / f"{model_key}_binary_confusion_matrix.png",
    )

    (report_dir / f"{model_key}_classification_report.txt").write_text(multi["report_text"], encoding="utf-8")
    (report_dir / f"{model_key}_binary_report.txt").write_text(
        "\n".join([
            f"{name} - tümör var/yok değerlendirmesi",
            "=" * 60,
            "Kural: no_tumor -> tümör yok (0); glioma / meningioma / pituitary -> tümör var (1)",
            "",
            f"Doğruluk:   {binary['accuracy']:.4f}",
            f"Kesinlik:   {binary['precision']:.4f}",
            f"Duyarlılık: {binary['recall']:.4f}",
            f"Özgüllük:   {binary['specificity']:.4f}",
            f"F1:         {binary['f1']:.4f}",
            "",
            binary["report_text"],
            f"Karışıklık matrisi (satır = gerçek, sütun = tahmin; sıra {BINARY_LABELS}):",
            str(binary["confusion_matrix"]),
            "",
        ]),
        encoding="utf-8",
    )
    return multi, binary


def evaluate_model(model_key: str) -> dict:
    """Kayıtlı modeli test setinde değerlendirir; rapor ve grafiklerini kaydeder."""
    from tensorflow.keras.models import load_model

    from src.data_loader import create_test_generator

    info = config.FINAL_MODELS[model_key]
    model_path = project_path(info["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    model = load_model(model_path)
    generator, test_df = create_test_generator(info["preprocess"])

    steps = math.ceil(generator.samples / config.BATCH_SIZE)
    probabilities = model.predict(generator, steps=steps, verbose=0)
    y_true = generator.classes.copy()
    y_pred = np.argmax(probabilities, axis=1)

    multi, binary = report_predictions(model_key, y_true, y_pred)
    plots.plot_misclassified_examples(
        test_df["filepath"].tolist(),
        [config.CLASS_NAMES[i] for i in y_true],
        [config.CLASS_NAMES[i] for i in y_pred],
        model_key, info["display_name"],
    )

    return {
        "key": model_key,
        "display_name": info["display_name"],
        "param_count": int(model.count_params()),
        "val_accuracy": best_validation_accuracy(model_key),
        "multi": multi,
        "binary": binary,
    }


def comparison_rows(results: list[dict]) -> list[dict]:
    """Karşılaştırma tablosu ve grafiği için sade satırlar."""
    return [
        {
            "key": r["key"],
            "display_name": r["display_name"],
            "val_accuracy": r["val_accuracy"],
            "test_accuracy": float(r["multi"]["accuracy"]),
            "macro_f1": float(r["multi"]["f1_macro"]),
            "binary_accuracy": float(r["binary"]["accuracy"]),
            "binary_recall": float(r["binary"]["recall"]),
            "binary_specificity": float(r["binary"]["specificity"]),
            "param_count": r["param_count"],
        }
        for r in results
    ]


def accuracy_standard_error(accuracy: float, n: int) -> float:
    """Tek bir test setinde ölçülen doğruluğun binom standart hatası."""
    return math.sqrt(accuracy * (1 - accuracy) / n)


def write_comparison_summary(results: list[dict], n_test: int) -> Path:
    rows = comparison_rows(results)
    header = (f"{'Model':<26} {'Doğr.(val)':>10} {'Doğr.(test)':>11} {'Makro F1':>9} "
              f"{'Var/yok':>8} {'Parametre':>11}")
    lines = ["Model karşılaştırması (test seti)", "=" * 60, f"Test örneği: {n_test}", "", header, "-" * len(header)]
    for row in rows:
        val = f"{row['val_accuracy']:.4f}" if row["val_accuracy"] is not None else "-"
        lines.append(
            f"{row['display_name']:<26} {val:>10} {row['test_accuracy']:>11.4f} {row['macro_f1']:>9.4f} "
            f"{row['binary_accuracy']:>8.4f} {row['param_count']:>11,}"
        )

    lines += ["", "En sık karıştırılan sınıflar (gerçek -> tahmin):"]
    for result in results:
        lines.append(f"  {result['display_name']}: "
                     f"{', '.join(most_confused_pairs(result['multi']['confusion_matrix']))}")

    errors = [accuracy_standard_error(row["test_accuracy"], n_test) for row in rows]
    lines += [
        "",
        f"Test doğruluğunun standart hatası yaklaşık ±{max(errors):.3f}'tür; modeller arasındaki "
        "bundan küçük farklar tek bir test setiyle ayırt edilemez.",
        "",
    ]

    report_dir = project_path(config.REPORT_DIR)
    ensure_dirs(report_dir)
    (report_dir / "test_metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    path = report_dir / "model_comparison_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_full_evaluation() -> list[dict]:
    """İki nihai modeli test setinde değerlendirir; karşılaştırma raporu ve grafiğini üretir."""
    results = [evaluate_model(key) for key in config.FINAL_MODELS]
    n_test = int(results[0]["multi"]["confusion_matrix"].sum())

    write_comparison_summary(results, n_test)
    plots.plot_model_comparison(
        [row for row in comparison_rows(results) if row["val_accuracy"] is not None],
        project_path(config.FIGURE_DIR) / "model_comparison.png",
    )
    return results
