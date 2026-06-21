# Deteksi Pola Aktivitas Penangkapan Ikan — Laut Banda

Aplikasi web untuk deploy model CNN deteksi aktivitas *fishing* berbasis data AIS
(Global Fishing Watch) di wilayah Laut Banda.

## Struktur Folder

```
tokoku_app/
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Dependency Python
├── model/
│   ├── model_cnn_laut_banda.keras
│   ├── scaler.pkl
│   ├── flag_encoder.pkl
│   └── gear_encoder.pkl
└── README.md
```

## Fitur

- **Input Manual**: form untuk prediksi satu titik data (lat/lon, bulan, jam, dll).
- **Upload CSV**: prediksi massal dari file CSV, lengkap dengan validasi kolom.
- **Peta interaktif**: visualisasi sebaran titik fishing vs non-fishing (pydeck).
- **Download hasil**: hasil prediksi CSV bisa diunduh kembali.

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Deploy ke Streamlit Community Cloud (Gratis)

1. Push folder ini ke repository GitHub (pastikan folder `model/` ikut ter-push;
   ukuran model `.keras` ~230 KB jadi aman untuk batas GitHub).
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **"New app"**, pilih repository, branch, dan set **Main file path** ke `app.py`.
4. Klik **Deploy**. Build pertama akan menginstall TensorFlow, jadi tunggu 3–5 menit.

### Catatan penting untuk deploy

- File `requirements.txt` memakai `tensorflow-cpu` (bukan `tensorflow` biasa) supaya
  build lebih ringan dan cepat di server Streamlit Cloud yang tidak punya GPU.
- Jika muncul error memory saat build, cek di **Settings → Advanced** apakah app
  berjalan di tier gratis (1 GB RAM) — model ini kecil (~50 ribu parameter) sehingga
  seharusnya aman.
- `scikit-learn` di-pin ke versi `1.6.1` agar identik dengan versi saat `scaler.pkl`
  dan encoder dibuat, menghindari `InconsistentVersionWarning` berubah jadi error.

## Format CSV untuk Upload Massal

Kolom wajib (nama harus persis sama):

| Kolom | Tipe | Keterangan |
|---|---|---|
| `month` | int | Bulan (1–12) |
| `day` | int | Isi `1` (kolom ini konstan pada data training) |
| `weekday` | int | 0 (Senin) – 6 (Minggu) |
| `cell_ll_lat` | float | Latitude, kisaran -8 s.d. -3 |
| `cell_ll_lon` | float | Longitude, kisaran 123 s.d. 132 |
| `flag` | string | Kode bendera kapal, contoh: `IDN`, `CHN`, `JPN` |
| `geartype` | string | Jenis alat tangkap, contoh: `fishing`, `purse_seines` |
| `hours` | float | Jam aktivitas terdeteksi |
| `mmsi_present` | int | Jumlah MMSI yang terdeteksi pada sel |

Template contoh bisa diunduh langsung dari tab **Upload CSV** di aplikasi.

## Catatan Model

- Arsitektur: `Conv1D(32) → MaxPool1D → Conv1D(64) → Flatten → Dense(64) → Dropout → Dense(1, sigmoid)`
- Input: 9 fitur numerik yang dinormalisasi dengan `StandardScaler`, di-reshape ke `(9, 1)`.
- Output: probabilitas sigmoid (0–1), ambang batas `0.5` untuk klasifikasi Fishing/Non-Fishing.
