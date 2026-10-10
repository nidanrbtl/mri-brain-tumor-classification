"""Bölme CSV dosyalarından Keras veri üreteçleri oluşturur."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.augmentation import get_train_augmentation_params
from src.preprocessing import get_preprocessor
from src.utils import get_project_root, project_path


def load_split_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """Bölme dosyasını okur; sütunları ve sınıf adlarını doğrular."""
    df = pd.read_csv(project_path(csv_path))

    if not {"filepath", "label"}.issubset(df.columns):
        raise ValueError(f"CSV'de 'filepath' ve 'label' sütunları olmalı: {csv_path}")

    unknown = set(df["label"].unique()) - set(config.CLASS_NAMES)
    if unknown:
        raise ValueError(f"Bilinmeyen etiketler: {sorted(unknown)}")
    return df


def get_dataset_sizes() -> dict[str, int]:
    return {
        "train": len(load_split_dataframe(config.TRAIN_CSV)),
        "val": len(load_split_dataframe(config.VAL_CSV)),
        "test": len(load_split_dataframe(config.TEST_CSV)),
    }


def _flow(datagen, dataframe: pd.DataFrame, shuffle: bool):
    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=str(get_project_root()),
        x_col="filepath",
        y_col="label",
        target_size=config.IMAGE_SIZE,
        classes=config.CLASS_NAMES,     # sınıf sırası sabit
        class_mode="categorical",
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.RANDOM_SEED if shuffle else None,
    )


def create_test_generator(preprocess: str):
    """Test seti üreteci: artırma yok, sıra sabit (tahminler etiketlerle eşleşsin diye)."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    test_df = load_split_dataframe(config.TEST_CSV)
    datagen = ImageDataGenerator(preprocessing_function=get_preprocessor(preprocess))
    return _flow(datagen, test_df, shuffle=False), test_df


def create_train_val_generators(preprocess: str) -> tuple:
    """Eğitim (artırmalı, karıştırılmış) ve doğrulama (artırmasız, sıralı) üreteçleri."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    preprocessing_function = get_preprocessor(preprocess)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        **get_train_augmentation_params(),
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_generator = _flow(train_datagen, load_split_dataframe(config.TRAIN_CSV), shuffle=True)
    val_generator = _flow(eval_datagen, load_split_dataframe(config.VAL_CSV), shuffle=False)
    return train_generator, val_generator
