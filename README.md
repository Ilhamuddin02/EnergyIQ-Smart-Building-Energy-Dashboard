# ⚡ EnergyIQ — Smart Building Energy Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> Proyek EAS · Mata Kuliah Pembelajaran Mesin  
> **Kelompok 14 · Kelas A**  
> Rafif Titan Athallah (2043231082) · Muhammad Ilhamuddin (2043231098)

---

## 📸 Tampilan Dashboard

| Tab | Deskripsi |
|-----|-----------|
| 📊 **Overview** | KPI utama, tren harian, pola jam & hari |
| 🔍 **EDA** | Distribusi, scatter, boxplot, heatmap korelasi |
| 🤖 **Model** | Metrik evaluasi, feature importance, simulasi skenario |
| 🏢 **Gedung** | Ranking konsumsi, proporsi per tipe, tabel detail |
| 🔮 **Prediksi** | Prediksi real-time, gauge, proyeksi 24 jam, rekomendasi |

---

## 🏗️ Arsitektur Proyek

```
EnergyIQ/
├── app.py                  ← Dashboard Streamlit (main)
├── requirements.txt        ← Dependensi Python
├── .streamlit/
│   └── config.toml         ← Tema & konfigurasi Streamlit
├── processed_data.csv      ← Dataset (20 gedung, ~8760 baris/gedung)
├── model_metadata.json     ← Metadata & metrik model
├── rf_model.pkl            ← Model Random Forest terlatih
├── le_building.pkl         ← LabelEncoder Building_ID
├── le_occupancy.pkl        ← LabelEncoder Occupancy_Level
└── le_type.pkl             ← LabelEncoder Building_Type
```

---

## 🤖 Model Machine Learning

| Komponen | Detail |
|----------|--------|
| Algoritma | **Random Forest Regressor** |
| n_estimators | 300 |
| max_depth | 15 |
| Split | Kronologis 80/20 (cegah data leakage) |
| Target | `Energy_Usage (kWh)` |

### 📊 Hasil Evaluasi

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **RMSE** | ~21.84 kWh | Rata-rata deviasi error |
| **MAE** | ~17.29 kWh | Error absolut rata-rata |
| **R²** | ~0.82 | 82% variansi terprediksi |
| **MAPE** | ~10.05% | Error relatif rata-rata |

### 🔑 Top Feature Importance

1. **Hour** (~27.6%) — pola puncak jam kerja
2. **Temperature (°C)** (~21.0%) — beban AC
3. **Day_sin** (~16.9%) — pola mingguan
4. **IsWeekend** (~9.1%) — penurunan weekend

---

## 🚀 Cara Menjalankan

### Lokal

```bash
# 1. Clone repo
git clone https://github.com/username/energyiq-dashboard.git
cd energyiq-dashboard

# 2. Install dependensi
pip install -r requirements.txt

# 3. Pastikan semua file model ada di direktori
# (rf_model.pkl, le_*.pkl, processed_data.csv, model_metadata.json)

# 4. Jalankan dashboard
streamlit run app.py
```

### Deploy ke Streamlit Cloud

1. Fork / push repo ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo & branch
4. Set **Main file path** = `app.py`
5. Klik **Deploy**

> **Catatan:** File `.pkl` harus ter-commit ke repo (perhatikan batas 100 MB GitHub).  
> Untuk file besar, gunakan [Git LFS](https://git-lfs.com/) atau simpan di Google Drive / Hugging Face Hub.

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
joblib>=1.3.0
```

---

## 💡 Fitur Dashboard

- **Filter interaktif** — tipe gedung, ID gedung, rentang tanggal
- **Prediksi real-time** — atur parameter di sidebar, hasil langsung tampil
- **Gauge meter** — visualisasi konsumsi vs baseline
- **Proyeksi 24 jam** — grafik prediksi per jam untuk hari/gedung yang dipilih
- **Rekomendasi otomatis** — saran operasional berdasarkan kondisi input
- **Dark theme** — tampilan profesional dengan palet warna konsisten

---

## 👥 Tim

| Nama | NIM |
|------|-----|
| Rafif Titan Athallah | 2043231082 |
| Muhammad Ilhamuddin | 2043231098 |

**Kelas A · Kelompok 14 · EAS Mata Kuliah Pembelajaran Mesin · 2025**

---

<p align="center">Made with ❤️ and ⚡ by Kelompok 14</p>
