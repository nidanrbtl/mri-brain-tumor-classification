"""
Veri hazırlığı: sınıf klasörlerini doğrular, bozuk görüntüleri ayıklar, veri
keşfi grafiklerini çizer ve katmanlı eğitim/doğrulama/test bölmesini kaydeder.

Kullanım: python scripts/prepare_data.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # proje kökü: config ve src için

import pandas as pd

import config
from src import plots
from src.dataset import class_counts, collect_image_records, stratified_split, validate_dataset_structure
from src.utils import ensure_dirs, project_path, set_seed


def write_dataset_summary(path: Path, valid: pd.DataFrame, n_corrupted: int, splits: dict[str, pd.DataFrame]) -> None:
    lines = [
        "Veri seti özeti",
        "=" * 60,
        f"Geçerli görüntü: {len(valid)}",
        f"Bozuk / okunamayan görüntü: {n_corrupted}",
        "",
        "Sınıf başına görüntü:",
        *[f"  {name}: {count}" for name, count in class_counts(valid).items()],
    ]
    for split_name, df in splits.items():
        lines += ["", f"{split_name} ({len(df)} görüntü):",
                  *[f"  {name}: {count}" for name, count in class_counts(df).items()]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="var olan bölme dosyalarının üzerine yaz")
    args = parser.parse_args()

    split_paths = {
        "Eğitim": project_path(config.TRAIN_CSV),
        "Doğrulama": project_path(config.VAL_CSV),
        "Test": project_path(config.TEST_CSV),
    }
    # Kayıtlı modeller bu bölmeyle eğitildi. Bölme değişirse eski test sonuçları geçersiz olur;
    # o yüzden üzerine yazmak açık bir istek gerektirir.
    if any(path.exists() for path in split_paths.values()) and not args.force:
        raise SystemExit("Bölme dosyaları zaten var. Yeniden üretmek için --force kullanın.")

    set_seed(config.RANDOM_SEED)
    data_dir = project_path(config.DATA_DIR)
    figure_dir, report_dir = project_path(config.FIGURE_DIR), project_path(config.REPORT_DIR)
    ensure_dirs(figure_dir, report_dir)

    validate_dataset_structure(data_dir)
    valid_records, corrupted_records = collect_image_records(data_dir)
    if not valid_records:
        raise SystemExit("Geçerli görüntü bulunamadı; veri seti yolunu kontrol edin.")
    print(f"Geçerli görüntü: {len(valid_records)} | Bozuk: {len(corrupted_records)}")

    valid = pd.DataFrame(valid_records)
    plots.plot_class_distribution(class_counts(valid), figure_dir / "class_distribution.png")
    plots.plot_sample_images(valid, figure_dir / "sample_images.png")

    splits = dict(zip(split_paths, stratified_split(valid_records)))
    for name, df in splits.items():
        df.to_csv(split_paths[name], index=False, encoding="utf-8")
        print(f"{name}: {len(df)} görüntü -> {split_paths[name].name}")

    if corrupted_records:
        pd.DataFrame(corrupted_records).to_csv(report_dir / "corrupted_images.csv", index=False, encoding="utf-8")
    write_dataset_summary(report_dir / "dataset_summary.txt", valid, len(corrupted_records), splits)


if __name__ == "__main__":
    main()
