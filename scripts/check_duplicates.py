"""
Bölmeler arasında içeriği aynı olan görüntü var mı diye bakar.

Aynı görüntü hem eğitim hem test kümesindeyse model onu "ezberden" tanır ve
test doğruluğu olduğundan yüksek çıkar. Bu script bayt bayt aynı dosyaları
bulur; yeniden boyutlandırılmış ya da aynı hastaya ait komşu kesitleri bulamaz.

Kullanım: python scripts/check_duplicates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # proje kökü: config ve src için

import config
from src.data_loader import load_split_dataframe
from src.dataset import find_duplicate_images
from src.utils import project_path


def main() -> None:
    splits = {
        "train": load_split_dataframe(config.TRAIN_CSV),
        "val": load_split_dataframe(config.VAL_CSV),
        "test": load_split_dataframe(config.TEST_CSV),
    }
    duplicates = find_duplicate_images(splits)
    total = sum(len(df) for df in splits.values())

    if duplicates.empty:
        print(f"{total} görüntü tarandı; içeriği aynı olan dosya bulunmadı.")
        return

    across = duplicates[duplicates["across_splits"]]
    test_hashes = set(across.loc[across["split"] == "test", "hash"])
    leaked_test = duplicates[(duplicates["split"] == "test") & duplicates["hash"].isin(test_hashes)]

    print(f"{total} görüntü tarandı.")
    print(f"Birden fazla kopyası olan görüntü: {duplicates['hash'].nunique()} ({len(duplicates)} dosya)")
    print(f"Kopyaları farklı kümelere düşen görüntü: {across['hash'].nunique()}")
    print(f"Başka bir kümede kopyası bulunan test görüntüsü: {len(leaked_test)} / {len(splits['test'])}")

    output = project_path(config.REPORT_DIR) / "duplicate_images.csv"
    duplicates.sort_values(["hash", "split"]).to_csv(output, index=False, encoding="utf-8")
    print(f"Liste: {output}")


if __name__ == "__main__":
    main()
