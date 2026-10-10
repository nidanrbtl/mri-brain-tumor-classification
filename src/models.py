"""Model mimarileri: sıfırdan eğitilen Custom CNN ve transfer öğrenmeli MobileNetV2."""

from __future__ import annotations

import io
from pathlib import Path

from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam

import config
from src.utils import ensure_dirs, project_path

INPUT_SHAPE = (*config.IMAGE_SIZE, 3)
NUM_CLASSES = len(config.CLASS_NAMES)

# MobileNetV2'nin üzerine eklenen sınıflandırıcı katmanların adları
HEAD_LAYER_NAMES = frozenset({"global_avg_pool", "dropout_head", "predictions"})


def _compile(model: Model, learning_rate: float) -> Model:
    """Bütün modeller aynı kayıp ve metrikle derlenir; karşılaştırma adil olsun diye."""
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def build_custom_cnn() -> Model:
    """
    Sıfırdan eğitilen evrişimli ağ (karşılaştırma tabanı).

    Dört evrişim bloğu (32, 64, 128, 256 filtre); her blokta BatchNormalization
    ve MaxPooling. Ardından GlobalAveragePooling ve 256 nöronlu bir Dense katman.
    """
    model = Sequential(name="custom_cnn")

    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for filters in (64, 128, 256):
        model.add(Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.3))
    # Flatten yerine GlobalAveragePooling: parametre sayısını ciddi biçimde azaltır
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    return _compile(model, config.TRAINING["custom_cnn"]["learning_rate"])


def build_mobilenetv2() -> Model:
    """
    ImageNet ağırlıklı MobileNetV2 tabanı + yeni sınıflandırıcı (özellik çıkarımı aşaması).
    Taban dondurulur; yalnızca üstteki sınıflandırıcı eğitilir.
    """
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False

    x = GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)
    x = Dropout(0.3, name="dropout_head")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="mobilenetv2_transfer")
    return _compile(model, config.TRAINING["mobilenetv2"]["learning_rate"])


def _is_batch_norm(layer) -> bool:
    name = layer.name.lower()
    return isinstance(layer, BatchNormalization) or "_bn" in name or name.endswith("bn") or "batchnorm" in name


def prepare_mobilenet_for_finetuning(model: Model) -> Model:
    """
    Özellik çıkarımıyla eğitilmiş modeli ince ayara hazırlar.

    Sınıflandırıcı ve tabanın son N katmanı eğitime açılır. BatchNormalization
    katmanları dondurulmuş kalır: küçük veriyle güncellenirlerse ImageNet'ten
    gelen istatistikler bozulur ve eğitim kararsızlaşır.
    """
    for layer in model.layers:
        layer.trainable = layer.name in HEAD_LAYER_NAMES

    base_layers = [
        layer for layer in model.layers
        if layer.name not in HEAD_LAYER_NAMES and layer.name != "input_layer"
    ]

    unfrozen = 0
    for layer in base_layers[-config.FINETUNE_UNFREEZE_LAYERS:]:
        if not _is_batch_norm(layer):
            layer.trainable = True
            unfrozen += 1

    if unfrozen == 0:
        raise ValueError("İnce ayar için açılabilecek taban katmanı bulunamadı.")

    # Düşük öğrenme oranı: ön eğitimli ağırlıklar yalnızca hafifçe güncellenmeli
    return _compile(model, config.TRAINING["mobilenetv2_finetuned"]["learning_rate"])


def count_parameters(model: Model) -> dict[str, int]:
    """Toplam, eğitilebilir ve dondurulmuş parametre sayıları."""
    from tensorflow.keras import backend

    trainable = int(sum(backend.count_params(w) for w in model.trainable_weights))
    total = int(model.count_params())
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def save_model_summary(model: Model, file_path: str | Path) -> Path:
    """model.summary() çıktısını metin dosyasına yazar."""
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))

    path = project_path(file_path)
    ensure_dirs(path.parent)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return path
