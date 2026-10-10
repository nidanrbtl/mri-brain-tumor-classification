"""Ham veri setini tarama, doğrulama, bölme ve yinelenen görüntü denetimi. TensorFlow gerektirmez."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.utils import CLASS_FOLDER_ALIASES, is_image_file, is_valid_image, project_path

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
SUBSET_FOLDERS = ("Training", "Testing", "train", "test", "val")


def resolve_class_directories(data_dir: Path, class_name: str) -> list[Path]:
    """
    Bir sınıfın görüntü klasörlerini bulur. Hem data/raw/<sınıf> hem de Kaggle'ın
    data/raw/Training/<sınıf> ve data/raw/Testing/<sınıf> düzeni desteklenir.
    """
    found: list[Path] = []
    for alias in CLASS_FOLDER_ALIASES.get(class_name, [class_name]):
        candidates = [data_dir / alias] + [data_dir / subset / alias for subset in SUBSET_FOLDERS]
        found.extend(directory for directory in candidates if directory.is_dir())

    unique, seen = [], set()
    for directory in found:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(directory)
    return unique


def logical_filepath(file_path: Path, class_dir: Path, data_dir: Path) -> str:
    """
    CSV'ye yazılacak, proje köküne göreli yol (data/raw/...).
    Veri klasörü bir bağlantı (junction) ile başka bir diskte duruyor olsa da yol aynı kalır.
    """
    relative = file_path.resolve().relative_to(class_dir.resolve())
    if class_dir.parent.resolve() == data_dir.resolve():
        return (Path(config.DATA_DIR) / class_dir.name / relative).as_posix()
    return (Path(config.DATA_DIR) / class_dir.parent.name / class_dir.name / relative).as_posix()


def collect_image_records(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Bütün sınıfların görüntülerini tarar; (geçerli, bozuk) kayıt listelerini döndürür."""
    valid, corrupted = [], []

    for class_name in config.CLASS_NAMES:
        seen: set[Path] = set()
        for class_dir in resolve_class_directories(data_dir, class_name):
            # sorted(): dosya sırası işletim sistemine göre değişmesin, bölme tekrarlanabilir olsun
            for file_path in sorted(class_dir.rglob("*")):
                resolved = file_path.resolve()
                if not file_path.is_file() or not is_image_file(file_path) or resolved in seen:
                    continue
                seen.add(resolved)
                record = {
                    "filepath": logical_filepath(file_path, class_dir, data_dir),
                    "label": class_name,
                    "filename": file_path.name,
                }
                (valid if is_valid_image(file_path) else corrupted).append(record)

    return valid, corrupted


def validate_dataset_structure(data_dir: Path) -> None:
    missing = [name for name in config.CLASS_NAMES if not resolve_class_directories(data_dir, name)]
    if missing:
        raise FileNotFoundError(
            f"Eksik sınıf klasörleri: {', '.join(missing)}\n"
            f"Veri seti '{data_dir}' altında Training/<sınıf> ve Testing/<sınıf> (ya da doğrudan <sınıf>) "
            "klasörleriyle bulunmalı. Bkz. data/README.md"
        )


def stratified_split(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """%70 eğitim, %15 doğrulama, %15 test; sınıf oranları üç kümede de korunur."""
    df = pd.DataFrame(records)
    train_df, rest_df = train_test_split(
        df, test_size=VAL_RATIO + TEST_RATIO, random_state=config.RANDOM_SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=config.RANDOM_SEED, stratify=rest_df["label"]
    )
    return tuple(part.reset_index(drop=True) for part in (train_df, val_df, test_df))


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts()
    return {name: int(counts.get(name, 0)) for name in config.CLASS_NAMES}


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def find_duplicate_images(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    İçeriği bayt bayt aynı olan görüntüleri bulur.

    Dönen tabloda her satır, birden fazla kez geçen bir görüntü kopyasıdır:
    hash, split, label, filepath ve kopyanın birden fazla kümeye yayılıp yayılmadığı
    (across_splits). Aynı görüntünün hem eğitim hem test kümesinde bulunması
    test başarısını olduğundan yüksek gösterir.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for split_name, df in splits.items():
        for row in df.itertuples(index=False):
            digest = file_hash(project_path(row.filepath))
            groups[digest].append(
                {"hash": digest, "split": split_name, "label": row.label, "filepath": row.filepath}
            )

    rows = []
    for members in groups.values():
        if len(members) > 1:
            across = len({member["split"] for member in members}) > 1
            rows.extend({**member, "across_splits": across} for member in members)
    return pd.DataFrame(rows, columns=["hash", "split", "label", "filepath", "across_splits"])
