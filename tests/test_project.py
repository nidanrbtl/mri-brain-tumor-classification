"""
TensorFlow ve veri seti gerektirmeyen testler: ön işleme, metrikler, veri
bölme, yinelenen görüntü denetimi ve eğitimin başlama/devam etme kuralları.

Çalıştırma: python -m unittest discover -s tests
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

import config
from src import dataset, evaluate, train
from src.predict import has_tumor, predict_image
from src.preprocessing import denoise_image, load_image, preprocess_image
from src.utils import TrainingLock


def random_image(seed: int, size=(300, 260), mode="L") -> Image.Image:
    rng = np.random.default_rng(seed)
    shape = (size[1], size[0]) if mode == "L" else (size[1], size[0], 3)
    return Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode)


def as_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class PreprocessingTests(unittest.TestCase):
    def test_load_image_matches_training_time_resize(self):
        image = random_image(1)
        loaded = load_image(image)
        self.assertEqual(loaded.shape, (*config.IMAGE_SIZE, 3))
        self.assertEqual(loaded.dtype, np.float32)
        # Keras'ın veri üreteciyle aynı yol: RGB'ye çevir, NEAREST ile küçült
        expected = np.asarray(image.convert("RGB").resize(config.IMAGE_SIZE[::-1], Image.NEAREST), dtype=np.float32)
        np.testing.assert_array_equal(loaded, expected)

    def test_bytes_path_and_pil_inputs_agree(self):
        image = random_image(2, mode="RGB")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mri.png"
            image.save(path)
            from_path = preprocess_image(path, "custom_cnn")
        np.testing.assert_array_equal(from_path, preprocess_image(as_png_bytes(image), "custom_cnn"))
        np.testing.assert_array_equal(from_path, preprocess_image(image, "custom_cnn"))

    def test_output_ranges(self):
        image = random_image(3)
        custom = preprocess_image(image, "custom_cnn")
        mobilenet = preprocess_image(image, "mobilenet")
        self.assertTrue(0.0 <= custom.min() and custom.max() <= 1.0)
        self.assertTrue(-1.0 <= mobilenet.min() and mobilenet.max() <= 1.0)
        # iki ölçekleme aynı görüntünün doğrusal dönüşümleridir
        np.testing.assert_allclose(mobilenet, custom * 2.0 - 1.0, atol=1e-6)

    def test_dark_image_is_not_rescaled(self):
        # Neredeyse siyah bir görüntü "zaten 0-1 aralığında" sanılıp 255 ile çarpılmamalı
        dark = Image.fromarray(np.ones((50, 50), dtype=np.uint8), "L")
        self.assertLess(preprocess_image(dark, "custom_cnn").max(), 0.01)

    def test_denoise_keeps_shape_and_range(self):
        image = load_image(random_image(4))
        denoised = denoise_image(image)
        self.assertEqual(denoised.shape, image.shape)
        self.assertTrue(0 <= denoised.min() and denoised.max() <= 255)
        self.assertLess(denoised.std(), image.std())  # bulanıklaştırma gürültüyü azaltır

    def test_unknown_preprocessor(self):
        with self.assertRaises(ValueError):
            preprocess_image(random_image(5), "resnet")


class PredictionTests(unittest.TestCase):
    class FakeModel:
        def __init__(self, probabilities):
            self.probabilities = np.array([probabilities], dtype=np.float32)
            self.seen_shape = None

        def predict(self, batch, verbose=0):
            self.seen_shape = batch.shape
            return self.probabilities

    def test_prediction_fields(self):
        model = self.FakeModel([0.05, 0.80, 0.10, 0.05])
        result = predict_image(model, random_image(6), "custom_cnn")
        self.assertEqual(model.seen_shape, (1, *config.IMAGE_SIZE, 3))
        self.assertEqual(result["predicted_class"], "meningioma")
        self.assertTrue(result["has_tumor"])
        self.assertAlmostEqual(result["confidence"], 0.80, places=5)
        self.assertEqual(list(result["probabilities"]), config.CLASS_NAMES)

    def test_no_tumor(self):
        result = predict_image(self.FakeModel([0.0, 0.1, 0.1, 0.8]), random_image(7), "mobilenet")
        self.assertEqual(result["predicted_class"], "no_tumor")
        self.assertFalse(result["has_tumor"])
        self.assertTrue(has_tumor("glioma"))


class MetricTests(unittest.TestCase):
    def setUp(self):
        #                 glioma  mening. pituit. no_tumor
        self.y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        self.y_pred = np.array([0, 1, 1, 3, 2, 2, 3, 0])

    def test_binary_mapping(self):
        np.testing.assert_array_equal(evaluate.to_binary([0, 1, 2, 3]), [1, 1, 1, 0])

    def test_multiclass_metrics(self):
        metrics = evaluate.compute_multiclass_metrics(self.y_true, self.y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 5 / 8)
        self.assertEqual(metrics["confusion_matrix"].tolist(),
                         [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 2, 0], [1, 0, 0, 1]])

    def test_binary_metrics(self):
        metrics = evaluate.compute_binary_metrics(self.y_true, self.y_pred)
        # tümör var: 6 örnek, biri kaçırıldı; tümör yok: 2 örnek, biri yanlış alarm
        self.assertEqual(metrics["confusion_matrix"].tolist(), [[1, 1], [1, 5]])
        self.assertAlmostEqual(metrics["recall"], 5 / 6)
        self.assertAlmostEqual(metrics["specificity"], 1 / 2)
        self.assertAlmostEqual(metrics["accuracy"], 6 / 8)

    def test_most_confused_pairs(self):
        cm = evaluate.compute_multiclass_metrics(self.y_true, self.y_pred)["confusion_matrix"]
        pairs = evaluate.most_confused_pairs(cm)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all("->" in pair for pair in pairs))

    def test_standard_error(self):
        self.assertAlmostEqual(evaluate.accuracy_standard_error(0.5, 100), 0.05)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._folder.name) / "raw"
        seed = 0
        # Kaggle düzeni: Training/<sınıf> ve Testing/<sınıf>; no_tumor klasörünün adı "notumor"
        for subset, count in (("Training", 14), ("Testing", 6)):
            for class_name in config.CLASS_NAMES:
                folder = self.data_dir / subset / ("notumor" if class_name == "no_tumor" else class_name)
                folder.mkdir(parents=True)
                for i in range(count):
                    seed += 1
                    random_image(seed, size=(40, 40)).save(folder / f"{subset[:2]}-{class_name[:2]}_{i}.jpg")

    def tearDown(self):
        self._folder.cleanup()

    def test_collect_and_split(self):
        (self.data_dir / "Training" / "glioma" / "bozuk.jpg").write_bytes(b"bu bir goruntu degil")
        (self.data_dir / "Training" / "glioma" / "notlar.txt").write_text("goruntu degil")

        dataset.validate_dataset_structure(self.data_dir)
        valid, corrupted = dataset.collect_image_records(self.data_dir)
        self.assertEqual(len(valid), 80)
        self.assertEqual([record["filename"] for record in corrupted], ["bozuk.jpg"])
        self.assertTrue(all(record["filepath"].startswith("data/raw/") for record in valid))
        self.assertEqual(dataset.class_counts(pd.DataFrame(valid)), {name: 20 for name in config.CLASS_NAMES})

        train_df, val_df, test_df = dataset.stratified_split(valid)
        self.assertEqual((len(train_df), len(val_df), len(test_df)), (56, 12, 12))
        self.assertEqual(dataset.class_counts(test_df), {name: 3 for name in config.CLASS_NAMES})
        paths = [set(df["filepath"]) for df in (train_df, val_df, test_df)]
        self.assertFalse(paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])

        # aynı girdiyle bölme her seferinde aynıdır
        again = dataset.stratified_split(valid)
        self.assertEqual(list(test_df["filepath"]), list(again[2]["filepath"]))

    def test_missing_class_folder(self):
        import shutil

        shutil.rmtree(self.data_dir / "Training" / "glioma")
        shutil.rmtree(self.data_dir / "Testing" / "glioma")
        with self.assertRaises(FileNotFoundError):
            dataset.validate_dataset_structure(self.data_dir)

    def test_duplicates_across_splits_are_found(self):
        original = self.data_dir / "Training" / "glioma" / "Tr-gl_0.jpg"
        copy = self.data_dir / "Testing" / "glioma" / "kopya.jpg"
        copy.write_bytes(original.read_bytes())
        other = self.data_dir / "Training" / "glioma" / "Tr-gl_1.jpg"

        frame = lambda *paths: pd.DataFrame({"filepath": [str(p) for p in paths], "label": "glioma"})
        duplicates = dataset.find_duplicate_images({"train": frame(original, other), "test": frame(copy)})
        self.assertEqual(len(duplicates), 2)
        self.assertTrue(duplicates["across_splits"].all())
        self.assertEqual(set(duplicates["split"]), {"train", "test"})

        self.assertTrue(dataset.find_duplicate_images({"train": frame(original, other)}).empty)


class TrainingRulesTests(unittest.TestCase):
    """Baştan başlama / devam etme kararı ve eşzamanlı eğitim kilidi."""

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        root = Path(self._folder.name)
        settings = dict(config.TRAINING["custom_cnn"], epochs=5,
                        best_path=str(root / "best.keras"), last_path=str(root / "last.keras"))
        self._patches = [
            mock.patch.object(config, "LOG_DIR", str(root / "logs")),
            mock.patch.object(config, "REPORT_DIR", str(root / "reports")),
            mock.patch.object(config, "MODEL_DIR", str(root / "models")),
            mock.patch.dict(config.TRAINING, {"custom_cnn": settings}),
        ]
        for patch in self._patches:
            patch.start()
        self.last_model = root / "last.keras"

    def tearDown(self):
        for patch in self._patches:
            patch.stop()
        self._folder.cleanup()

    def write_history(self, val_accuracies):
        path = train.history_path("custom_cnn")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "epoch": range(len(val_accuracies)), "accuracy": 0.5, "learning_rate": 5e-4, "loss": 1.0,
            "val_accuracy": val_accuracies, "val_loss": 1.0,
        }).to_csv(path, index=False)

    def test_first_run_starts_fresh(self):
        start = train.resolve_start_point("custom_cnn", resume=False, fresh=False)
        self.assertEqual((start.completed_epochs, start.best_val_accuracy, start.resuming), (0, None, False))

    def test_existing_records_require_an_explicit_choice(self):
        self.write_history([0.6, 0.8])
        with self.assertRaises(RuntimeError):
            train.resolve_start_point("custom_cnn", resume=False, fresh=False)
        self.assertFalse(train.resolve_start_point("custom_cnn", resume=False, fresh=True).resuming)

    def test_resume_keeps_the_best_score_so_far(self):
        self.write_history([0.6, 0.9, 0.7])
        self.last_model.write_bytes(b"checkpoint")
        start = train.resolve_start_point("custom_cnn", resume=True, fresh=False)
        self.assertEqual(start.completed_epochs, 3)
        # devam eden oturum, 0.9'u geçmeden "en iyi" modelin üzerine yazamaz
        self.assertAlmostEqual(start.best_val_accuracy, 0.9)

    def test_resume_needs_records_and_a_checkpoint(self):
        with self.assertRaises(RuntimeError):
            train.resolve_start_point("custom_cnn", resume=True, fresh=False)  # kayıt yok
        self.write_history([0.6])
        with self.assertRaises(RuntimeError):
            train.resolve_start_point("custom_cnn", resume=True, fresh=False)  # checkpoint yok

    def test_finished_training_cannot_be_resumed(self):
        self.write_history([0.5, 0.6, 0.7, 0.8, 0.9])
        self.last_model.write_bytes(b"checkpoint")
        with self.assertRaises(RuntimeError):
            train.resolve_start_point("custom_cnn", resume=True, fresh=False)

    def test_conflicting_flags(self):
        with self.assertRaises(ValueError):
            train.resolve_start_point("custom_cnn", resume=True, fresh=True)

    def test_fresh_start_clears_old_records(self):
        self.write_history([0.6])
        train.clear_previous_records("custom_cnn")
        self.assertTrue(train.read_history("custom_cnn").empty)

    def test_second_training_process_is_refused(self):
        with TrainingLock("custom_cnn"):
            with self.assertRaises(RuntimeError):
                with TrainingLock("custom_cnn"):
                    pass
        with TrainingLock("custom_cnn"):  # ilk kilit bırakıldıktan sonra yeniden alınabilir
            pass

    def test_lock_is_released_after_an_error(self):
        with self.assertRaises(KeyError):
            with TrainingLock("custom_cnn"):
                raise KeyError("eğitim sırasında hata")
        with TrainingLock("custom_cnn"):
            pass


if __name__ == "__main__":
    unittest.main()
