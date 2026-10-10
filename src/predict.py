"""Tek görüntü için tahmin: dört sınıflı sonuç ve ondan türetilen tümör var/yok bilgisi."""

from __future__ import annotations

import numpy as np

import config
from src.preprocessing import preprocess_image

CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Hipofiz tümörü",
    "no_tumor": "Tümör yok",
}


def has_tumor(predicted_class: str) -> bool:
    return predicted_class != config.NO_TUMOR_CLASS


def predict_image(model, source, preprocess: str) -> dict:
    """
    Görüntüyü (dosya yolu, bayt dizisi ya da PIL görüntüsü) ön işler ve sınıflandırır.
    Dönen sözlük: tahmin edilen sınıf, güven değeri, sınıf olasılıkları, tümör var/yok.
    """
    batch = np.expand_dims(preprocess_image(source, preprocess), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_class = config.CLASS_NAMES[int(np.argmax(probabilities))]

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_DISPLAY_NAMES[predicted_class],
        "confidence": float(np.max(probabilities)),
        "probabilities": {name: float(p) for name, p in zip(config.CLASS_NAMES, probabilities)},
        "has_tumor": has_tumor(predicted_class),
    }
