"""İki nihai modeli test setinde değerlendirir. Kullanım: python scripts/run_evaluation.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # proje kökü: config ve src için

import config
from src.evaluate import accuracy_standard_error, comparison_rows, run_full_evaluation
from src.utils import set_seed


def main() -> None:
    set_seed(config.RANDOM_SEED)
    results = run_full_evaluation()
    n_test = int(results[0]["multi"]["confusion_matrix"].sum())

    print(f"Test seti: {config.TEST_CSV} ({n_test} görüntü)\n")
    for row in comparison_rows(results):
        error = accuracy_standard_error(row["test_accuracy"], n_test)
        print(row["display_name"])
        print(f"  Test doğruluğu:          {row['test_accuracy']:.4f} (±{error:.3f})")
        print(f"  Makro F1:                {row['macro_f1']:.4f}")
        print(f"  Tümör var/yok doğruluğu: {row['binary_accuracy']:.4f} "
              f"(duyarlılık {row['binary_recall']:.4f}, özgüllük {row['binary_specificity']:.4f})")
    print(f"\nRaporlar: {config.REPORT_DIR} | Grafikler: {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
