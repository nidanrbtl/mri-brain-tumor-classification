# Veri

Veri seti bu depoda yer almaz. Kaggle'daki **Brain Tumor MRI Dataset** indirilip
`data/raw/` altına şu düzenle açılmalıdır:

```
data/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Projede kullanılan sürümde 7.200 görüntü vardır: her sınıfta 1.800 (Training
altında 1.400, Testing altında 400).

Hangi görüntünün eğitim, doğrulama ve test kümesine düştüğü
`outputs/reports/train_split.csv`, `val_split.csv` ve `test_split.csv`
dosyalarında kayıtlıdır. Kayıtlı modeller bu bölmeyle eğitildiği için sonuçları
yeniden üretmek isteyenlerin bu dosyaları değiştirmemesi gerekir;
`scripts/prepare_data.py` var olan bölmenin üzerine `--force` verilmeden yazmaz.
