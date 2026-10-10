"""
Streamlit arayüzü: MRI görüntüsü yükleme, dört sınıflı tahmin ve tümör var/yok sonucu.

Çalıştırma: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # config ve src için

import pandas as pd
import streamlit as st

import config
from src.predict import CLASS_DISPLAY_NAMES, predict_image

METRICS_PATH = PROJECT_ROOT / config.REPORT_DIR / "test_metrics.json"


@st.cache_resource
def load_model_cached(model_path: str):
    """Modeli bir kez yükler; sonraki tahminlerde bellekten kullanılır."""
    from tensorflow.keras.models import load_model

    full_path = PROJECT_ROOT / model_path
    return load_model(str(full_path)) if full_path.exists() else None


@st.cache_data
def load_test_metrics() -> dict[str, dict]:
    """run_evaluation.py'nin kaydettiği test metrikleri (dosya yoksa boş sözlük)."""
    if not METRICS_PATH.exists():
        return {}
    return {row["key"]: row for row in json.loads(METRICS_PATH.read_text(encoding="utf-8"))}


def show_prediction(result: dict) -> None:
    left, right = st.columns(2)

    with left:
        st.metric("Tahmin edilen sınıf", result["predicted_label"])
        st.metric("Modelin güveni", f"%{result['confidence'] * 100:.1f}")
        if result["has_tumor"]:
            st.error("Tümör var")
        else:
            st.success("Tümör yok")

    with right:
        chart = pd.DataFrame(
            {
                "Sınıf": [CLASS_DISPLAY_NAMES[name] for name in config.CLASS_NAMES],
                "Olasılık (%)": [result["probabilities"][name] * 100 for name in config.CLASS_NAMES],
            }
        ).set_index("Sınıf")
        st.bar_chart(chart)


def show_model_performance(metrics: dict[str, dict]) -> None:
    if not metrics:
        return
    st.markdown("### Test seti başarımı")
    for column, (key, info) in zip(st.columns(len(config.FINAL_MODELS)), config.FINAL_MODELS.items()):
        row = metrics.get(key)
        if row is None:
            continue
        with column:
            st.markdown(f"**{info['display_name']}**")
            st.write(f"- Doğruluk (4 sınıf): %{row['test_accuracy'] * 100:.1f}")
            st.write(f"- Makro F1: {row['macro_f1']:.3f}")
            st.write(f"- Tümör var/yok doğruluğu: %{row['binary_accuracy'] * 100:.1f}")


def main() -> None:
    st.set_page_config(page_title="MRI Beyin Tümörü Sınıflandırma", page_icon="🧠", layout="wide")

    st.title("MRI Görüntülerinden Beyin Tümörü Sınıflandırma")
    st.caption("Nida Nur Batal · Örüntü Tanıma dönem projesi")
    st.warning(
        "Bu uygulama bir ders projesidir. Tıbbi cihaz değildir; tanı ya da tedavi kararı için kullanılamaz."
    )

    left, right = st.columns(2)

    with left:
        model_key = st.selectbox(
            "Model",
            options=list(config.FINAL_MODELS),
            format_func=lambda key: config.FINAL_MODELS[key]["display_name"],
        )
        model_info = config.FINAL_MODELS[model_key]
        model = load_model_cached(model_info["path"])
        if model is None:
            st.error(f"Model dosyası bulunamadı: `{model_info['path']}`")
            st.stop()

        uploaded_file = st.file_uploader("MRI görüntüsü (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    with right:
        if uploaded_file is None:
            st.info("Tahmin için bir MRI görüntüsü yükleyin.")
        else:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### Tahmin")
        try:
            with st.spinner("Tahmin yapılıyor..."):
                result = predict_image(model, uploaded_file.getvalue(), model_info["preprocess"])
        except Exception as error:  # bozuk ya da görüntü olmayan dosya
            st.error(f"Görüntü okunamadı: {error}")
        else:
            show_prediction(result)

    st.markdown("---")
    show_model_performance(load_test_metrics())
    st.caption(
        "Veri: Kaggle Brain Tumor MRI Dataset · 7.200 görüntü · "
        "sınıflar: glioma, meningioma, hipofiz tümörü, tümör yok"
    )


if __name__ == "__main__":
    main()
