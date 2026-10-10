"""
MobileNetV2, 2. aşama (ince ayar): 1. aşamanın en iyi modelinden başlar, tabanın
son katmanlarını düşük öğrenme oranıyla eğitir.
Kullanım: python scripts/finetune_mobilenet.py [--resume | --fresh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # proje kökü: config ve src için

import config
from src.train import add_training_arguments, read_history, run_training
from src.utils import project_path


def build_finetuning_model():
    from tensorflow.keras.models import load_model

    from src.models import prepare_mobilenet_for_finetuning

    source = project_path(config.TRAINING["mobilenetv2"]["best_path"])
    if not source.exists():
        raise FileNotFoundError(
            f"1. aşama modeli bulunamadı: {source}\nÖnce scripts/train_mobilenet.py çalıştırılmalı."
        )
    return prepare_mobilenet_for_finetuning(load_model(source))


def compare_with_feature_extraction(history) -> list[str]:
    """Özet rapora, ince ayarın 1. aşamaya göre ne kazandırdığını ekler."""
    stage_one = read_history("mobilenetv2")
    if stage_one.empty:
        return []
    before, after = stage_one["val_accuracy"].max(), history["val_accuracy"].max()
    return [
        "1. aşama (özellik çıkarımı) ile karşılaştırma:",
        f"  Doğrulama doğruluğu: {before:.4f} -> {after:.4f} ({after - before:+.4f})",
        f"  Açılan taban katmanı: son {config.FINETUNE_UNFREEZE_LAYERS} katman (BatchNormalization hariç)",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    args = parser.parse_args()

    run_training(
        "mobilenetv2_finetuned", "mobilenet", build_finetuning_model,
        resume=args.resume, fresh=args.fresh, extra_summary=compare_with_feature_extraction,
    )


if __name__ == "__main__":
    main()
