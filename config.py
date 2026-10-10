"""
Proje genelinde kullanılan sabitler ve dosya yolları.
Bütün modüller ayarları buradan okur; aynı değer iki yerde tanımlanmaz.
"""

# Görüntü ve eğitim parametreleri
IMAGE_SIZE = (224, 224)   # (yükseklik, genişlik)
BATCH_SIZE = 32
RANDOM_SEED = 42

# Sınıf sırası modelin çıkış sırasıdır; değiştirilirse kayıtlı modeller geçersiz olur.
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
NO_TUMOR_CLASS = "no_tumor"

# Dizinler (proje köküne göre)
DATA_DIR = "data/raw"
MODEL_DIR = "models"
FIGURE_DIR = "outputs/figures"
LOG_DIR = "outputs/logs"
REPORT_DIR = "outputs/reports"

# prepare_data.py'nin ürettiği bölme dosyaları (filepath, label)
TRAIN_CSV = "outputs/reports/train_split.csv"
VAL_CSV = "outputs/reports/val_split.csv"
TEST_CSV = "outputs/reports/test_split.csv"

# Her eğitim aşamasının ayarları. "best": doğrulama doğruluğu en yüksek model,
# "last": son tamamlanan epoch (kesilen eğitime devam etmek için).
TRAINING = {
    "custom_cnn": {
        "title": "Custom CNN",
        "epochs": 25,
        "learning_rate": 5e-4,
        "best_path": "models/custom_cnn_best.keras",
        "last_path": "models/custom_cnn_last.keras",
    },
    "mobilenetv2": {
        "title": "MobileNetV2 (özellik çıkarımı)",
        "epochs": 15,
        "learning_rate": 3e-4,
        "best_path": "models/mobilenetv2_best.keras",
        "last_path": "models/mobilenetv2_last.keras",
    },
    "mobilenetv2_finetuned": {
        "title": "MobileNetV2 (ince ayar)",
        "epochs": 10,
        "learning_rate": 1e-5,
        "best_path": "models/mobilenetv2_finetuned_best.keras",
        "last_path": "models/mobilenetv2_finetuned_last.keras",
    },
}

# İnce ayarda MobileNetV2 tabanının son kaç katmanının eğitime açılacağı
FINETUNE_UNFREEZE_LAYERS = 25

# Test setinde değerlendirilen ve arayüzde sunulan iki nihai model
FINAL_MODELS = {
    "custom_cnn": {
        "display_name": "Custom CNN",
        "path": TRAINING["custom_cnn"]["best_path"],
        "preprocess": "custom_cnn",
    },
    "mobilenetv2_finetuned": {
        "display_name": "MobileNetV2 Fine-Tuned",
        "path": TRAINING["mobilenetv2_finetuned"]["best_path"],
        "preprocess": "mobilenet",
    },
}
