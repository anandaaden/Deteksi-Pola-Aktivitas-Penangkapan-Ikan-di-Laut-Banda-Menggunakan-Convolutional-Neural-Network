import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import pydeck as pdk
from pathlib import Path

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Deteksi Pola Penangkapan Ikan - Laut Banda",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PALETTE DEEP OCEAN 
# ============================================================
OCEAN = {
    'deep': '#0B132B',      # Hitam kebiruan (deep sea)
    'dark': '#1C2541',      # Biru dongker
    'mid': '#3A506B',       # Biru abu
    'light': '#5BC0BE',     # Turquoise
    'white': '#FFFFFF',     # Putih
    'gold': '#FFC857',      # Kuning emas (untuk highlight)
    'coral': '#FF6B6B',     # Merah karang (untuk fishing)
}

# ============================================================
# CUSTOM CSS DEEP OCEAN THEME (tanpa mengubah logika)
# ============================================================
st.markdown(f"""
<style>
    /* Global background */
    .stApp {{
        background: linear-gradient(135deg, {OCEAN['deep']} 0%, {OCEAN['dark']} 50%, {OCEAN['mid']} 100%);
    }}
    
    /* Main content area */
    .main .block-container {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem 3rem;
        border: 1px solid rgba(91, 192, 190, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {OCEAN['light']} !important;
        text-shadow: 0 2px 10px rgba(91, 192, 190, 0.3);
        font-weight: 600 !important;
    }}
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {{
        background: rgba(11, 19, 43, 0.95) !important;
        border-right: 2px solid {OCEAN['light']} !important;
        backdrop-filter: blur(20px);
    }}
    
    /* Sidebar text */
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span {{
        color: {OCEAN['white']} !important;
    }}
    
    .css-1d391kg .stMarkdown {{
        color: {OCEAN['white']} !important;
    }}
    
    /* Sidebar title */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {{
        color: {OCEAN['light']} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {OCEAN['light']} 0%, {OCEAN['mid']} 100%) !important;
        color: {OCEAN['deep']} !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(91, 192, 190, 0.3) !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(91, 192, 190, 0.5) !important;
        background: linear-gradient(135deg, {OCEAN['light']} 0%, {OCEAN['gold']} 100%) !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background: rgba(11, 19, 43, 0.5);
        border-radius: 15px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(91, 192, 190, 0.2);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {OCEAN['white']} !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: rgba(91, 192, 190, 0.2) !important;
        color: {OCEAN['light']} !important;
        box-shadow: 0 0 20px rgba(91, 192, 190, 0.2) !important;
    }}
    
    /* Metrics */
    .stMetric {{
        background: rgba(11, 19, 43, 0.6) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        border: 1px solid rgba(91, 192, 190, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    .stMetric label {{
        color: {OCEAN['light']} !important;
        font-weight: 500 !important;
    }}
    
    .stMetric value {{
        color: {OCEAN['white']} !important;
        font-size: 2rem !important;
    }}
    
    /* Dataframe */
    .stDataFrame {{
        background: rgba(11, 19, 43, 0.6) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(91, 192, 190, 0.1) !important;
    }}
    
    .stDataFrame table {{
        color: {OCEAN['white']} !important;
    }}
    
    .stDataFrame thead th {{
        background: rgba(91, 192, 190, 0.2) !important;
        color: {OCEAN['light']} !important;
    }}
    
    /* Select boxes, sliders, inputs */
    .stSelectbox, .stSlider, .stNumberInput {{
        background: rgba(11, 19, 43, 0.5) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(91, 192, 190, 0.2) !important;
    }}
    
    .stSelectbox label, .stSlider label, .stNumberInput label {{
        color: {OCEAN['light']} !important;
        font-weight: 500 !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {OCEAN['deep']}, {OCEAN['light']}) !important;
        border-radius: 10px !important;
    }}
    
    /* Success/Info messages */
    .stAlert {{
        background: rgba(11, 19, 43, 0.8) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(91, 192, 190, 0.3) !important;
        color: {OCEAN['white']} !important;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background: rgba(11, 19, 43, 0.5) !important;
        border-radius: 10px !important;
        color: {OCEAN['light']} !important;
        border: 1px solid rgba(91, 192, 190, 0.2) !important;
    }}
    
    .streamlit-expanderContent {{
        background: rgba(11, 19, 43, 0.3) !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid rgba(91, 192, 190, 0.1) !important;
        border-top: none !important;
    }}
    
    /* Captions and footers */
    .stCaption, footer {{
        color: rgba(255, 255, 255, 0.6) !important;
    }}
    
    /* Divider */
    hr {{
        border-color: rgba(91, 192, 190, 0.3) !important;
        margin: 2rem 0 !important;
    }}
    
    /* Download buttons */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, {OCEAN['mid']} 0%, {OCEAN['dark']} 100%) !important;
        color: {OCEAN['white']} !important;
        border: 1px solid {OCEAN['light']} !important;
        border-radius: 25px !important;
        transition: all 0.3s ease !important;
    }}
    
    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, {OCEAN['light']} 0%, {OCEAN['mid']} 100%) !important;
        color: {OCEAN['deep']} !important;
        transform: translateY(-2px) !important;
    }}
    
    /* Markdown text */
    .stMarkdown p, .stMarkdown li {{
        color: rgba(255, 255, 255, 0.9) !important;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        background: {OCEAN['deep']};
    }}
    
    ::-webkit-scrollbar-track {{
        background: {OCEAN['deep']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {OCEAN['light']}, {OCEAN['mid']});
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {OCEAN['gold']};
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL DIRECTORY & KONSTANTA
# ============================================================
MODEL_DIR = Path(__file__).parent / "model"

LAT_MIN, LAT_MAX = -8.0, -3.0
LON_MIN, LON_MAX = 123.0, 132.0

FEATURE_ORDER = [
    "month", "day", "weekday", "cell_ll_lat", "cell_ll_lon",
    "flag", "geartype", "hours", "mmsi_present",
]


# ============================================================
# LOAD MODEL & PREPROCESSOR (cached)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_DIR / "model_cnn_laut_banda.keras")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    flag_encoder = joblib.load(MODEL_DIR / "flag_encoder.pkl")
    gear_encoder = joblib.load(MODEL_DIR / "gear_encoder.pkl")
    return model, scaler, flag_encoder, gear_encoder


model, scaler, flag_encoder, gear_encoder = load_artifacts()


# ============================================================
# FUNGSI PREDIKSI (TIDAK BERUBAH)
# ============================================================
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    df harus memiliki kolom mentah:
    month, day, weekday, cell_ll_lat, cell_ll_lon, flag, geartype, hours, mmsi_present
    flag & geartype dalam bentuk string (label asli), akan di-encode di sini.
    """
    work = df.copy()

    # Encode kategorikal, tangani label tak dikenal
    def safe_encode(encoder, series, col_name):
        known = set(encoder.classes_)
        unknown_mask = ~series.isin(known)
        if unknown_mask.any():
            bad_vals = series[unknown_mask].unique().tolist()
            raise ValueError(
                f"Nilai tidak dikenal pada kolom '{col_name}': {bad_vals}. "
                f"Nilai yang valid: {list(encoder.classes_)}"
            )
        return encoder.transform(series)

    work["flag_enc"] = safe_encode(flag_encoder, work["flag"].astype(str), "flag")
    work["geartype_enc"] = safe_encode(gear_encoder, work["geartype"].astype(str), "geartype")

    feat_df = pd.DataFrame({
        "month": work["month"],
        "day": work["day"],
        "weekday": work["weekday"],
        "cell_ll_lat": work["cell_ll_lat"],
        "cell_ll_lon": work["cell_ll_lon"],
        "flag": work["flag_enc"],
        "geartype": work["geartype_enc"],
        "hours": work["hours"],
        "mmsi_present": work["mmsi_present"],
    })[FEATURE_ORDER]

    X_scaled = scaler.transform(feat_df)
    X_input = X_scaled.reshape(-1, 9, 1)

    probs = model.predict(X_input, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    result = df.copy()
    result["probabilitas_fishing"] = probs
    result["prediksi"] = np.where(preds == 1, "Fishing", "Non-Fishing")
    return result


# ============================================================
# SIDEBAR (TIDAK BERUBAH - HANYA TAMBAH CSS DI ATAS)
# ============================================================
with st.sidebar:
    st.title("🌊 Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini menampilkan model **CNN (Convolutional Neural Network)**
        untuk mendeteksi pola aktivitas penangkapan ikan di **Laut Banda**
        menggunakan data **AIS (Automatic Identification System)** dari
        **Global Fishing Watch (GFW)**.

        """
    )
    st.divider()
    st.markdown("** Cakupan wilayah model**")
    st.markdown(f"- Lintang: `{LAT_MIN}°` s.d. `{LAT_MAX}°`")
    st.markdown(f"- Bujur: `{LON_MIN}°` s.d. `{LON_MAX}°`")
    st.divider()
    
    st.caption(" Input: 9 fitur ternormalisasi (StandardScaler)")


# ============================================================
# HEADER (TAMBAH EMOJI & STYLING)
# ============================================================
st.title("🌊 Deteksi Pola Aktivitas Penangkapan Ikan — Laut Banda")
st.markdown(
    """
    <div style='background: rgba(91, 192, 190, 0.1); padding: 1rem; border-radius: 15px; 
         border-left: 4px solid #5BC0BE; margin-bottom: 2rem;'>
        <p style='color: #FFFFFF; margin: 0;'>
            Prediksi apakah suatu titik aktivitas kapal di Laut Banda tergolong 
            <strong style='color: #5BC0BE;'>aktivitas penangkapan ikan (fishing)</strong> 
            atau <strong style='color: #FF6B6B;'>bukan</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tab_form, tab_csv = st.tabs([" Input Manual", " Upload CSV"])


# ============================================================
# TAB 1: INPUT MANUAL (TIDAK BERUBAH)
# ============================================================
with tab_form:
    st.subheader(" Input Data Satu Titik")
    st.caption("Isi parameter di bawah ini sesuai data AIS yang ingin diuji.")

    col1, col2, col3 = st.columns(3)

    with col1:
        month = st.slider(
            " Bulan",
            1, 12, 6,
            help="Bulan saat aktivitas kapal terjadi (1 = Januari, 12 = Desember). "
                 "Model belajar pola musiman, misalnya musim angin timur vs barat.",
        )
        weekday = st.selectbox(
            " Hari (weekday)",
            options=list(range(7)),
            format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x],
            index=1,
            help="Hari dalam seminggu saat aktivitas terjadi. Beberapa alat tangkap "
                 "punya pola hari operasi yang khas (misalnya jeda di akhir pekan).",
        )
        hours = st.number_input(
            " Jam aktivitas (hours)",
            min_value=0.0, max_value=24.0, value=4.0, step=0.5,
            help="Total durasi (dalam jam) kapal terdeteksi aktif/diam di sel grid ini "
                 "selama satu hari, berdasarkan sinyal AIS. Semakin lama menetap di satu "
                 "titik, semakin besar indikasi aktivitas menangkap ikan.",
        )

    with col2:
        lat = st.slider(
            " Latitude (cell_ll_lat)",
            LAT_MIN, LAT_MAX, -5.5, step=0.05,
            help="Garis lintang sudut kiri-bawah sel grid AIS (satuan derajat). "
                 "Nilai negatif berarti di selatan garis khatulistiwa.",
        )
        lon = st.slider(
            " Longitude (cell_ll_lon)",
            LON_MIN, LON_MAX, 127.5, step=0.05,
            help="Garis bujur sudut kiri-bawah sel grid AIS (satuan derajat), "
                 "menunjukkan posisi timur-barat di wilayah Laut Banda.",
        )
        mmsi_present = st.number_input(
            " MMSI present",
            min_value=0.0, max_value=5.0, value=1.0, step=1.0,
            help="Jumlah kapal unik (berdasarkan nomor identitas MMSI) yang terdeteksi "
                 "berada di sel grid ini. Lebih dari 1 berarti beberapa kapal beroperasi "
                 "berdekatan di titik yang sama.",
        )

    with col3:
        flag = st.selectbox(
            " Bendera kapal (flag)",
            options=list(flag_encoder.classes_),
            index=list(flag_encoder.classes_).index("IDN") if "IDN" in flag_encoder.classes_ else 0,
            help="Negara bendera (flag state) kapal, yaitu negara tempat kapal "
                 "terdaftar secara hukum — bukan selalu negara asal pemiliknya.",
        )
        geartype = st.selectbox(
            " Jenis alat tangkap (geartype)",
            options=list(gear_encoder.classes_),
            help="Jenis alat tangkap yang diidentifikasi GFW dari pola gerak kapal, "
                 "contoh: 'drifting_longlines' (pancing rawai hanyut), "
                 "'purse_seines' (pukat cincin), 'squid_jigger' (pancing cumi).",
        )
        st.markdown("&nbsp;")

    if st.button(" Prediksi", type="primary", use_container_width=False):
        input_df = pd.DataFrame([{
            "month": month,
            "day": 1,
            "weekday": weekday,
            "cell_ll_lat": lat,
            "cell_ll_lon": lon,
            "flag": flag,
            "geartype": geartype,
            "hours": hours,
            "mmsi_present": mmsi_present,
        }])

        try:
            result = predict_batch(input_df)
            pred_label = result["prediksi"].iloc[0]
            prob = result["probabilitas_fishing"].iloc[0]

            st.divider()
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if pred_label == "Fishing":
                    st.success(f"###  Hasil: {pred_label}")
                else:
                    st.info(f"###  Hasil: {pred_label}")
                st.metric(
                    " Probabilitas Fishing",
                    f"{prob:.2%}",
                    help="Keluaran model (0-100%). Di atas 50% diklasifikasikan sebagai "
                         "'Fishing', di bawah 50% sebagai 'Non-Fishing'.",
                )
                st.progress(float(prob))

            with res_col2:
                # MAP dengan warna disesuaikan tema deep ocean
                map_df = pd.DataFrame([{
                    "lat": lat, "lon": lon,
                    "label": pred_label,
                    "color": [255, 107, 107, 200] if pred_label == "Fishing" else [91, 192, 190, 200],
                }])
                view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=6, pitch=0)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius=15000,
                    pickable=True,
                )
                st.pydeck_chart(pdk.Deck(
                    map_style="dark",  # DARK MAP untuk tema deep ocean
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": "{label}"},
                ))

        except ValueError as e:
            st.error(f"❌ Terjadi kesalahan input: {e}")


# ============================================================
# TAB 2: UPLOAD CSV (TIDAK BERUBAH - HANYA GANTI WARNA MAP)
# ============================================================
with tab_csv:
    st.subheader(" Prediksi Massal via CSV")
    st.markdown(
        "Unggah file CSV dengan kolom berikut (nama harus persis sama):"
    )

    feature_desc = pd.DataFrame([
        {"Kolom": "month", "Tipe": "int", "Keterangan": "Bulan aktivitas (1=Januari ... 12=Desember). Menangkap pola musiman."},
        {"Kolom": "day", "Tipe": "int", "Keterangan": "Tanggal dalam bulan. Pada data training nilainya konstan (1), isi dengan 1 jika tidak yakin."},
        {"Kolom": "weekday", "Tipe": "int", "Keterangan": "Hari dalam seminggu, 0=Senin ... 6=Minggu."},
        {"Kolom": "cell_ll_lat", "Tipe": "float", "Keterangan": f"Latitude sudut kiri-bawah sel grid AIS, kisaran {LAT_MIN}° s.d. {LAT_MAX}°."},
        {"Kolom": "cell_ll_lon", "Tipe": "float", "Keterangan": f"Longitude sudut kiri-bawah sel grid AIS, kisaran {LON_MIN}° s.d. {LON_MAX}°."},
        {"Kolom": "flag", "Tipe": "string", "Keterangan": "Negara bendera kapal (flag state), contoh: IDN, CHN, JPN. Lihat daftar lengkap di bawah."},
        {"Kolom": "geartype", "Tipe": "string", "Keterangan": "Jenis alat tangkap, contoh: fishing, purse_seines, drifting_longlines. Lihat daftar lengkap di bawah."},
        {"Kolom": "hours", "Tipe": "float", "Keterangan": "Total jam aktivitas kapal terdeteksi di sel grid tersebut dalam satu hari."},
        {"Kolom": "mmsi_present", "Tipe": "int", "Keterangan": "Jumlah kapal unik (MMSI) yang terdeteksi di sel grid tersebut."},
    ])
    st.dataframe(feature_desc, use_container_width=True, hide_index=True)

    st.caption(
        "Kolom `flag` dan `geartype` diisi label asli (contoh: `IDN`, `fishing`), "
        "bukan angka hasil encoding."
    )

    with st.expander(" Lihat nilai valid untuk flag & geartype"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("** flag**")
            st.write(list(flag_encoder.classes_))
        with c2:
            st.markdown("** geartype**")
            st.write(list(gear_encoder.classes_))

    sample_csv = pd.DataFrame([
        {"month": 6, "day": 1, "weekday": 2, "cell_ll_lat": -5.5, "cell_ll_lon": 127.5,
         "flag": "IDN", "geartype": "fishing", "hours": 5.0, "mmsi_present": 1},
        {"month": 8, "day": 1, "weekday": 4, "cell_ll_lat": -6.2, "cell_ll_lon": 129.1,
         "flag": "CHN", "geartype": "purse_seines", "hours": 12.0, "mmsi_present": 2},
    ])
    st.download_button(
        " Unduh Contoh Template CSV",
        data=sample_csv.to_csv(index=False).encode("utf-8"),
        file_name="template_prediksi_laut_banda.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(" Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in FEATURE_ORDER if c not in raw_df.columns]
            if missing_cols:
                st.error(f" Kolom berikut tidak ditemukan di CSV: {missing_cols}")
            else:
                with st.spinner(" Memproses prediksi..."):
                    result = predict_batch(raw_df)

                st.success(f" Berhasil memproses {len(result)} baris data.")

                m1, m2, m3 = st.columns(3)
                m1.metric(" Total Titik", len(result))
                m2.metric(" Terdeteksi Fishing", int((result["prediksi"] == "Fishing").sum()))
                m3.metric(" Non-Fishing", int((result["prediksi"] == "Non-Fishing").sum()))

                st.divider()
                st.subheader(" Peta Sebaran Hasil Prediksi")

                # MAP DENGAN WARNA DEEP OCEAN
                map_data = result.copy()
                map_data["color"] = map_data["prediksi"].apply(
                    lambda x: [255, 107, 107, 180] if x == "Fishing" else [91, 192, 190, 180]
                )
                view_state = pdk.ViewState(
                    latitude=float(map_data["cell_ll_lat"].mean()),
                    longitude=float(map_data["cell_ll_lon"].mean()),
                    zoom=5.5,
                    pitch=0,
                )
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position="[cell_ll_lon, cell_ll_lat]",
                    get_fill_color="color",
                    get_radius=8000,
                    pickable=True,
                )
                st.pydeck_chart(pdk.Deck(
                    map_style="dark",  # DARK MAP untuk tema deep ocean
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": "{prediksi}\nProbabilitas: {probabilitas_fishing:.2%}"},
                ))

                # LEGEND dengan warna deep ocean
                legend_col1, legend_col2 = st.columns(2)
                legend_col1.markdown("🔴 **Fishing**")
                legend_col2.markdown("🔵 **Non-Fishing**")

                st.divider()
                st.subheader(" Tabel Hasil")
                st.dataframe(result, use_container_width=True)

                st.download_button(
                    " Unduh Hasil Prediksi (CSV)",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_laut_banda.csv",
                    mime="text/csv",
                )

        except ValueError as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")


# ============================================================
# FOOTER (TIDAK BERUBAH)
# ============================================================
st.divider()
st.caption(
    "🌊 Model CNN dilatih menggunakan data AIS Global Fishing Watch (GFW) wilayah Laut Banda. "
    "Hasil prediksi bersifat estimatif. "
    "\n\n Kelompok 11 Machine Learning"
)