"""Custom CNN eğitimi. Kullanım: python scripts/train_custom_cnn.py [--resume | --fresh]"""

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

    from src.models import build_custom_cnn

    run_training("custom_cnn", "custom_cnn", build_custom_cnn, resume=args.resume, fresh=args.fresh)


if __name__ == "__main__":
    main()
