"""
Ortak eğitim akışı: baştan başlama ya da kaldığı yerden devam, checkpoint'ler,
epoch kayıtları, eğitim eğrileri ve özet rapor.

İki kural bu modülün varlık sebebidir:

* Aynı model için aynı anda yalnızca bir eğitim süreci çalışabilir (TrainingLock).
* Devam eden bir eğitim, daha önce kaydedilmiş daha iyi bir modelin üzerine
  yazamaz: "en iyi" checkpoint'i, önceki oturumların en yüksek doğrulama
  doğruluğu geçilmeden güncellenmez.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from src.plots import plot_training_history
from src.utils import StreamTee, TrainingLock, ensure_dirs, project_path, set_seed


def history_path(model_name: str) -> Path:
    return project_path(config.LOG_DIR) / f"{model_name}_history.csv"


def epoch_log_path(model_name: str) -> Path:
    return project_path(config.REPORT_DIR) / f"{model_name}_epoch_log.txt"


def read_history(model_name: str) -> pd.DataFrame:
    """Epoch başına metrikler (her epoch sonunda diske yazılır). Kayıt yoksa boş tablo."""
    path = history_path(model_name)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
    return pd.read_csv(path)


@dataclass
class StartPoint:
    """Eğitimin hangi epoch'tan ve hangi 'en iyi' değerle başlayacağı."""

    completed_epochs: int = 0
    best_val_accuracy: float | None = None

    @property
    def resuming(self) -> bool:
        return self.completed_epochs > 0


def resolve_start_point(model_name: str, resume: bool, fresh: bool) -> StartPoint:
    """
    Önceki eğitim kayıtlarına ve kullanıcının isteğine göre başlangıç noktasını belirler.

    Kayıt varken ne yapılacağı tahmin edilmez; kullanıcı --resume ya da --fresh
    ile açıkça söylemek zorundadır.
    """
    if resume and fresh:
        raise ValueError("--resume ve --fresh birlikte kullanılamaz.")

    settings = config.TRAINING[model_name]
    history = read_history(model_name)
    last_model = project_path(settings["last_path"])

    if history.empty:
        if resume:
            raise RuntimeError(f"Devam edilecek eğitim kaydı yok: {history_path(model_name)}")
        return StartPoint()

    if fresh:
        return StartPoint()

    if not resume:
        raise RuntimeError(
            f"'{model_name}' için önceki bir eğitimin kayıtları var ({len(history)} epoch).\n"
            "Kaldığı yerden sürdürmek için --resume, kayıtları silip baştan başlamak için --fresh kullanın."
        )

    if not last_model.exists():
        raise RuntimeError(f"Devam etmek için son checkpoint gerekli ama bulunamadı: {last_model}")
    if len(history) >= settings["epochs"]:
        raise RuntimeError(
            f"Bu eğitim zaten tamamlanmış ({len(history)}/{settings['epochs']} epoch). "
            "Yeniden eğitmek için --fresh kullanın."
        )
    return StartPoint(completed_epochs=len(history), best_val_accuracy=float(history["val_accuracy"].max()))


def clear_previous_records(model_name: str) -> None:
    """Baştan başlarken önceki eğitimin epoch kayıtlarını siler (modeller checkpoint ile yenilenir)."""
    for path in (history_path(model_name), epoch_log_path(model_name)):
        path.unlink(missing_ok=True)


def create_callbacks(model_name: str, start: StartPoint) -> list:
    """Checkpoint, erken durdurma, öğrenme oranı düşürme ve kayıt callback'leri."""
    from tensorflow.keras.callbacks import (
        Callback,
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    settings = config.TRAINING[model_name]
    ensure_dirs(project_path(config.MODEL_DIR), project_path(config.LOG_DIR), project_path(config.REPORT_DIR))
    log_path = epoch_log_path(model_name)

    class EpochFileLogger(Callback):
        """Her epoch sonunda metrikleri okunabilir bir metin dosyasına ekler."""

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            line = (
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Epoch {epoch + 1} | "
                f"loss={logs.get('loss', 0):.4f} | accuracy={logs.get('accuracy', 0):.4f} | "
                f"val_loss={logs.get('val_loss', 0):.4f} | val_accuracy={logs.get('val_accuracy', 0):.4f}\n"
            )
            with log_path.open("a", encoding="utf-8") as file:
                file.write(line)

    return [
        # En iyi model: doğrulama doğruluğu, bugüne kadarki en iyi değeri geçerse kaydedilir.
        # initial_value_threshold olmadan, devam eden oturum ilk epoch'unda dosyanın üzerine yazar.
        ModelCheckpoint(
            filepath=str(project_path(settings["best_path"])),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            initial_value_threshold=start.best_val_accuracy,
            verbose=1,
        ),
        # Son model: her epoch sonunda; kesilen eğitime devam etmek için.
        ModelCheckpoint(filepath=str(project_path(settings["last_path"])), save_best_only=False, verbose=0),
        EpochFileLogger(),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        CSVLogger(filename=str(history_path(model_name)), append=start.resuming),
    ]


def write_training_summary(
    model_name: str,
    history: pd.DataFrame,
    started: datetime,
    finished: datetime,
    start: StartPoint,
    parameters: dict[str, int],
    extra_lines: list[str] | None = None,
) -> Path:
    """Eğitimin kısa özetini metin dosyasına yazar."""
    settings = config.TRAINING[model_name]
    best_row = history.loc[history["val_accuracy"].idxmax()]
    duration_minutes = (finished - started).total_seconds() / 60

    lines = [
        f"{settings['title']} - eğitim özeti",
        "=" * 60,
        f"Oturum başlangıcı:  {started:%Y-%m-%d %H:%M:%S}",
        f"Oturum bitişi:      {finished:%Y-%m-%d %H:%M:%S}  ({duration_minutes:.1f} dakika)",
        f"Devam edilen epoch: {start.completed_epochs}",
        f"Tamamlanan epoch:   {len(history)} / {settings['epochs']}",
        "",
        f"En iyi doğrulama doğruluğu: {best_row['val_accuracy']:.4f} (epoch {int(best_row['epoch']) + 1})",
        f"En düşük doğrulama kaybı:   {history['val_loss'].min():.4f}",
        "",
        f"Parametre sayısı: {parameters['total']:,} "
        f"(eğitilebilir {parameters['trainable']:,}, dondurulmuş {parameters['frozen']:,})",
        f"Öğrenme oranı: {settings['learning_rate']} | Batch: {config.BATCH_SIZE} | Tohum: {config.RANDOM_SEED}",
        "Callback'ler: EarlyStopping(val_loss, patience=5), ReduceLROnPlateau(0.2, patience=3)",
        "",
        f"En iyi model: {settings['best_path']}",
        f"Son model:    {settings['last_path']}",
    ]
    if extra_lines:
        lines += ["", *extra_lines]

    path = project_path(config.REPORT_DIR) / f"{model_name}_training_summary.txt"
    ensure_dirs(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_training(
    model_name: str,
    preprocess: str,
    build_model: Callable[[], object],
    resume: bool = False,
    fresh: bool = False,
    extra_summary: Callable[[pd.DataFrame], list[str]] | None = None,
) -> pd.DataFrame:
    """
    Bir modelin eğitim oturumunu baştan sona yürütür ve bütün oturumların
    birleşik epoch geçmişini döndürür.

    build_model yalnızca baştan başlanırken çağrılır; devam ederken son checkpoint yüklenir.
    """
    from tensorflow.keras.models import load_model

    from src.data_loader import create_train_val_generators
    from src.models import count_parameters

    settings = config.TRAINING[model_name]

    with TrainingLock(model_name):
        start = resolve_start_point(model_name, resume=resume, fresh=fresh)
        if not start.resuming:
            clear_previous_records(model_name)

        console_log = project_path(config.REPORT_DIR) / f"{model_name}_console_log.txt"
        tee = StreamTee(sys.__stdout__, console_log, title=settings["title"], mode="a" if start.resuming else "w")
        original_stdout, sys.stdout = sys.stdout, tee
        try:
            set_seed(config.RANDOM_SEED)
            train_generator, val_generator = create_train_val_generators(preprocess)

            if start.resuming:
                print(f"Epoch {start.completed_epochs} tamamlanmış; kaldığı yerden devam ediliyor "
                      f"(şimdiye kadarki en iyi doğrulama doğruluğu {start.best_val_accuracy:.4f}).")
                model = load_model(project_path(settings["last_path"]))
            else:
                print("Eğitim baştan başlıyor.")
                model = build_model()

            parameters = count_parameters(model)
            print(f"Eğitim: {train_generator.samples} | Doğrulama: {val_generator.samples} | "
                  f"Parametre: {parameters['total']:,} (eğitilebilir {parameters['trainable']:,})")

            started = datetime.now()
            model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=settings["epochs"],
                initial_epoch=start.completed_epochs,
                steps_per_epoch=math.ceil(train_generator.samples / config.BATCH_SIZE),
                validation_steps=math.ceil(val_generator.samples / config.BATCH_SIZE),
                callbacks=create_callbacks(model_name, start),
                verbose=1,
            )
            finished = datetime.now()

            # CSVLogger her epoch'u diske yazar; dosya bütün oturumları içerir.
            history = read_history(model_name)
            plot_training_history(history, model_name, settings["title"])
            summary_path = write_training_summary(
                model_name, history, started, finished, start, parameters,
                extra_lines=extra_summary(history) if extra_summary else None,
            )
            print(f"\nEğitim tamamlandı. En iyi doğrulama doğruluğu: {history['val_accuracy'].max():.4f}")
            print(f"Özet: {summary_path}")
            return history
        finally:
            sys.stdout = original_stdout
            tee.close()


def add_training_arguments(parser) -> None:
    """Eğitim scriptlerinin ortak komut satırı seçenekleri."""
    parser.add_argument("--resume", action="store_true", help="kesilen eğitime son checkpoint'ten devam et")
    parser.add_argument("--fresh", action="store_true", help="önceki kayıtları silip baştan eğit")
