"""Yardımcı fonksiyonlar: tohum sabitleme, yollar, görüntü doğrulama, loglama, eğitim kilidi."""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import config

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Kaggle veri setinde "no_tumor" sınıfının klasörü "notumor" adıyla gelir.
CLASS_FOLDER_ALIASES = {
    "no_tumor": ["no_tumor", "notumor", "no tumor"],
}


def get_project_root() -> Path:
    """config.py'nin bulunduğu klasör."""
    return Path(__file__).resolve().parent.parent


def project_path(relative: str | Path) -> Path:
    """Göreli bir yolu proje köküne göre mutlak yola çevirir."""
    path = Path(relative)
    return path if path.is_absolute() else get_project_root() / path


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Python, NumPy ve (kuruluysa) TensorFlow rastgele sayı üreteçlerini sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_valid_image(file_path: Path) -> bool:
    """Dosya açılıp piksel verisi okunabiliyorsa True döner; bozuk dosyalar bölmeden önce elenir."""
    if not is_image_file(file_path):
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()  # dosya bütünlüğü
        with Image.open(file_path) as img:  # verify() sonrası dosya yeniden açılmalıdır
            img.load()
            return img.size[0] > 0 and img.size[1] > 0
    except Exception:
        return False


class StreamTee:
    """
    Yazılanı hem konsola hem bir log dosyasına gönderir.

    Keras ilerleme çubuğu akıştan ``encoding`` gibi özellikler bekler; bilinmeyen
    özellikler gerçek akışa yönlendirilir.
    """

    def __init__(self, stream, log_path: Path, title: str, mode: str = "w"):
        self._stream = stream
        self.encoding = getattr(stream, "encoding", None) or "utf-8"
        self.errors = getattr(stream, "errors", "strict")
        ensure_dirs(log_path.parent)
        self._log_file = log_path.open(mode, encoding="utf-8")
        self._log_file.write(f"{title} - {datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 60}\n")

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def write(self, message: str) -> int:
        self._stream.write(message)
        self._log_file.write(message)
        return len(message)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        self._log_file.close()


class TrainingLock:
    """
    Aynı modelin iki ayrı süreçte aynı anda eğitilmesini engeller.

    İki eğitim süreci aynı checkpoint dosyalarına yazarsa, biri diğerinin kaydettiği
    daha iyi modelin üzerine yazabilir. Kilit dosyası varken ikinci süreç başlamaz.
    """

    def __init__(self, model_name: str):
        self.path = project_path(config.MODEL_DIR) / f".{model_name}.lock"
        self._held = False

    def __enter__(self) -> "TrainingLock":
        ensure_dirs(self.path.parent)
        try:
            # "x" kipi: dosya zaten varsa hata verir; kontrol ve oluşturma tek adımdır.
            with self.path.open("x", encoding="utf-8") as file:
                file.write(f"pid={os.getpid()} baslangic={datetime.now():%Y-%m-%d %H:%M:%S}\n")
        except FileExistsError:
            raise RuntimeError(
                f"Bu model için başka bir eğitim sürüyor görünüyor: {self.path}\n"
                "Gerçekten çalışan bir eğitim yoksa (örneğin önceki süreç çöktüyse) "
                "bu dosyayı silip yeniden deneyin."
            ) from None
        self._held = True
        return self

    def __exit__(self, *exc_info) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False
