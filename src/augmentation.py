"""Veri artırma ayarları. Yalnızca eğitim setine uygulanır."""

from __future__ import annotations


def get_train_augmentation_params() -> dict:
    """
    Keras ImageDataGenerator için artırma parametreleri.
    Değerler bilerek hafif tutulmuştur; güçlü dönüşümler anatomik yapıyı bozabilir.
    """
    return {
        "rotation_range": 15,              # kesit açısındaki küçük farklar
        "zoom_range": 0.10,                # farklı görüntüleme ölçekleri
        "horizontal_flip": True,           # beynin sol-sağ simetrisi
        "brightness_range": (0.90, 1.10),  # kontrast farklılıkları
        "width_shift_range": 0.05,
        "height_shift_range": 0.05,
        "fill_mode": "nearest",
    }
