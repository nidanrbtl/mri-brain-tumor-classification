"""
Görüntü ön işleme: yükleme, boyutlandırma, gürültü azaltma ve ölçekleme.

Eğitim, değerlendirme ve arayüzdeki tahmin aynı adımları kullanır. Bir modelin
ölçülen başarısı ancak tahmin sırasında da eğitimdeki ön işleme uygulanırsa geçerlidir.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config
from src.utils import project_path


def load_image(source: str | Path | bytes | Image.Image) -> np.ndarray:
    """
    Görüntüyü RGB olarak yükler ve modelin giriş boyutuna getirir (float32, 0-255).

    Eğitimde Keras'ın veri üreteci görüntüleri "nearest" enterpolasyonla
    küçültür. Tek görüntü tahmininde de aynı yöntem kullanılır; farklı bir
    enterpolasyon, modele eğitimde görmediği türden pikseller verir.
    """
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, bytes):
        image = Image.open(BytesIO(source))
    else:
        image = Image.open(project_path(source))

    height, width = config.IMAGE_SIZE
    resized = image.convert("RGB").resize((width, height), Image.NEAREST)
    return np.asarray(resized, dtype=np.float32)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    3x3 Gauss bulanıklaştırma ile hafif gürültü azaltma (0-255 ölçeğinde girdi bekler).
    Çekirdek küçük tutulur ki tümör sınırları silinmesin.
    """
    as_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(as_uint8, (3, 3), sigmaX=0.8).astype(np.float32)


def scale_for_custom_cnn(image: np.ndarray) -> np.ndarray:
    """0-255 piksel değerlerini 0-1 aralığına getirir."""
    return np.clip(image / 255.0, 0.0, 1.0).astype(np.float32)


def scale_for_mobilenet(image: np.ndarray) -> np.ndarray:
    """
    0-255 piksel değerlerini -1 ile 1 arasına getirir.
    ``keras.applications.mobilenet_v2.preprocess_input`` ile aynı işlemdir
    (x / 127.5 - 1); ön eğitimli ağırlıklar bu ölçekle eğitilmiştir.
    """
    return (image.astype(np.float32) / 127.5) - 1.0


# Veri üreteçlerinin her görüntüye uyguladığı fonksiyonlar (girdi: 0-255, float32)
def preprocess_array_for_custom_cnn(image: np.ndarray) -> np.ndarray:
    return scale_for_custom_cnn(denoise_image(image))


def preprocess_array_for_mobilenet(image: np.ndarray) -> np.ndarray:
    return scale_for_mobilenet(denoise_image(image))


PREPROCESSORS = {
    "custom_cnn": preprocess_array_for_custom_cnn,
    "mobilenet": preprocess_array_for_mobilenet,
}


def get_preprocessor(name: str):
    if name not in PREPROCESSORS:
        raise ValueError(f"Desteklenmeyen ön işleme türü: {name}. Seçenekler: {sorted(PREPROCESSORS)}")
    return PREPROCESSORS[name]


def preprocess_image(source: str | Path | bytes | Image.Image, preprocess: str) -> np.ndarray:
    """Tek bir görüntüyü (dosya yolu, bayt dizisi ya da PIL görüntüsü) modele hazır hâle getirir."""
    return get_preprocessor(preprocess)(load_image(source))
