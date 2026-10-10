"""
MobileNetV2, 1. aşama (özellik çıkarımı): ImageNet tabanı dondurulur, yalnızca
sınıflandırıcı eğitilir. Kullanım: python scripts/train_mobilenet.py [--resume | --fresh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # proje kökü: config ve src için

from src.train import add_training_arguments, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    args = parser.parse_args()

    from src.models import build_mobilenetv2

    run_training("mobilenetv2", "mobilenet", build_mobilenetv2, resume=args.resume, fresh=args.fresh)


if __name__ == "__main__":
    main()
