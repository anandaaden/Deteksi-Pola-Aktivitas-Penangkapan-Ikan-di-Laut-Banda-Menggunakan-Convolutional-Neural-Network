import streamlit as st
import numpy as np
import pandas as pd
import base64
import joblib
import tensorflow as tf
import folium
from streamlit_folium import st_folium
from pathlib import Path
import io

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="MataBanda — CNN Fish Predictor",
    page_icon="MataBanda",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path(__file__).parent / "model"
LAT_MIN, LAT_MAX = -8.0, -3.0
LON_MIN, LON_MAX = 123.0, 132.0
FEATURE_ORDER = ["month", "day", "weekday", "cell_ll_lat", "cell_ll_lon",
                 "flag", "geartype", "hours", "mmsi_present"]

FLAGS_LABEL = {
    "AUS": "AUS (Australia)", "CHN": "CHN (Tiongkok)", "ESP": "ESP (Spanyol)",
    "HKG": "HKG (Hong Kong)", "IDN": "IDN (Indonesia)", "JPN": "JPN (Jepang)",
    "KOR": "KOR (Korea)", "NOR": "NOR (Norwegia)", "NZL": "NZL (Selandia Baru)",
    "PAN": "PAN (Panama)", "UNKNOWN-AUS": "UNKNOWN-AUS",
}
GEAR_LABEL = {
    "drifting_longlines": "Drifting Longlines (Rawai Hanyut)",
    "fishing": "Fishing (Umum)",
    "other_purse_seines": "Other Purse Seines",
    "pole_and_line": "Pole and Line (Pancing Tonda)",
    "purse_seines": "Purse Seines (Pukat Cincin)",
    "set_gillnets": "Set Gillnets (Jaring Insang)",
    "set_longlines": "Set Longlines (Rawai Dasar)",
    "squid_jigger": "Squid Jigger (Pancing Cumi)",
    "trawlers": "Trawlers (Pukat Hela)",
    "trollers": "Trollers (Pancing Trolling)",
    "tuna_purse_seines": "Tuna Purse Seines (Pukat Tuna)",
}
VALID_FLAGS = list(FLAGS_LABEL.keys())
VALID_GEARS = list(GEAR_LABEL.keys())

# ============================================================
# CUSTOM CSS — MataBanda palette
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

:root {
    --drone-bg:       #EDF3FB;
    --drone-white:    #FFFFFF;
    --drone-lblue:    #CBE3EF;
    --drone-blue:     #4636FC;
    --drone-dark:     #043D70;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--drone-bg) !important;
    color: var(--drone-dark) !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* Promo banner */
.bw-promo {
    background: linear-gradient(90deg, #4636FC, #3A4163);
    color: white;
    text-align: center;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 8px 16px;
    letter-spacing: 0.01em;
}

/* Navbar */
.bw-nav {
    background: white;
    border-bottom: 1px solid var(--drone-lblue);
    padding: 0 2rem;
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
}
.bw-logo-box {
    background: var(--drone-blue);
    color: white;
    padding: 8px 10px;
    border-radius: 12px;
    font-size: 1.1rem;
    margin-right: 10px;
    display: inline-flex;
}
.bw-brand { font-size: 1.25rem; font-weight: 800; color: var(--drone-dark); letter-spacing: -0.02em; }
.bw-brand span { color: var(--drone-blue); }
.bw-subbrand { font-size: 0.6rem; color: #9ca3af; letter-spacing: 0.15em; text-transform: uppercase; display: block; margin-top: -2px; }
.bw-status {
    display: inline-flex; align-items: center; gap: 6px;
    background: #d1fae5; color: #065f46;
    font-size: 0.72rem; font-weight: 700;
    padding: 4px 12px; border-radius: 999px;
}
.bw-dot {
    width: 8px; height: 8px;
    background: #10b981; border-radius: 50%;
    animation: ping 1.2s cubic-bezier(0,0,0.2,1) infinite;
}
@keyframes ping {
    0%,100%{opacity:1;transform:scale(1)}
    50%{opacity:.6;transform:scale(1.4)}
}

/* Page wrapper */
.bw-page { max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem; }

/* Cards */
.bw-card {
    background: white;
    border: 1px solid rgba(203,227,239,.6);
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 4px rgba(58,65,99,.04);
}
.bw-card-title {
    font-size: 1rem; font-weight: 700; color: var(--drone-dark);
    display: flex; align-items: center; gap: 8px; margin-bottom: 1rem;
}
.bw-icon { color: var(--drone-blue); }

/* Hero */
.bw-hero {
    background: linear-gradient(135deg, #fff 0%, #EDF3FB 60%, rgba(203,227,239,.4) 100%);
    border: 1px solid rgba(203,227,239,.6);
    border-radius: 24px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.25rem;
}
.bw-hero-badge {
    font-size: 0.7rem; font-weight: 700; color: var(--drone-blue);
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 8px;
}
.bw-hero h1 {
    font-size: 1.9rem; font-weight: 800; color: var(--drone-dark);
    line-height: 1.2; margin-bottom: .75rem; letter-spacing: -0.02em;
}
.bw-hero p { font-size: .9rem; color: #4b5563; line-height: 1.65; max-width: 580px; }
.bw-ship-bg {
    position: absolute; right: -20px; bottom: -20px;
    font-size: 200px; color: var(--drone-dark); opacity: .05;
    pointer-events: none;
}

/* Tab pills */
.bw-tabs {
    display: flex; gap: 6px;
    background: rgba(255,255,255,.85); border: 1px solid var(--drone-lblue);
    border-radius: 16px; padding: 6px; width: fit-content; margin-top: 1.25rem;
}
.bw-tab {
    padding: 8px 20px; border-radius: 12px;
    font-size: .82rem; font-weight: 700; cursor: pointer; border: none;
    transition: all .2s;
}
.bw-tab-active { background: var(--drone-blue); color: white; box-shadow: 0 2px 8px rgba(90,168,214,.3); }
.bw-tab-inactive { background: transparent; color: rgba(58,65,99,.6); }
.bw-tab-inactive:hover { color: var(--drone-dark); }

/* Timeline arsitektur */
.bw-timeline { position: relative; padding-left: 24px; }
.bw-timeline::before {
    content: ''; position: absolute; left: 6px; top: 10px; bottom: 0;
    width: 2px; background: var(--drone-lblue);
}
.bw-tl-item { position: relative; margin-bottom: 16px; }
.bw-tl-dot {
    position: absolute; left: -22px; top: 4px;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--drone-blue); border: 3px solid white;
    box-shadow: 0 0 0 2px var(--drone-lblue);
}
.bw-tl-dot.dark { background: var(--drone-dark); }
.bw-tl-title { font-size: .82rem; font-weight: 700; color: var(--drone-dark); }
.bw-tl-desc { font-size: .73rem; color: #6b7280; line-height: 1.45; margin-top: 2px; }

/* Encoder badge chips */
.bw-chip {
    display: inline-block;
    background: var(--drone-bg); border: 1px solid var(--drone-lblue);
    border-radius: 6px; padding: 2px 8px;
    font-size: .7rem; font-family: monospace; font-weight: 600;
    color: var(--drone-dark); margin: 2px;
}

/* Result cards */
.bw-result-fishing {
    background: #fff1f2; border: 1px solid #fecdd3; border-radius: 16px; padding: 1rem;
}
.bw-result-nonfishing {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 16px; padding: 1rem;
}
.bw-result-label { font-size: 1.4rem; font-weight: 800; margin-top: 4px; }
.bw-result-desc { font-size: .7rem; margin-top: 4px; }
.bw-result-fishing .bw-result-label { color: #9f1239; }
.bw-result-fishing .bw-result-desc { color: rgba(159,18,57,.7); }
.bw-result-nonfishing .bw-result-label { color: #065f46; }
.bw-result-nonfishing .bw-result-desc { color: rgba(6,95,70,.7); }

.bw-prob-card {
    background: rgba(237,243,251,.6); border: 1px solid var(--drone-lblue);
    border-radius: 16px; padding: 1rem;
}
.bw-prob-label { font-size: .7rem; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }
.bw-prob-value { font-size: 2rem; font-weight: 900; color: var(--drone-dark); margin: 4px 0; }
.bw-progress-bar { background: #e5e7eb; height: 6px; border-radius: 999px; overflow: hidden; }
.bw-progress-fill { background: var(--drone-blue); height: 100%; transition: width .4s; }

.bw-coord-card {
    background: rgba(237,243,251,.6); border: 1px solid var(--drone-lblue);
    border-radius: 16px; padding: 1rem;
}
.bw-coord-label { font-size: .7rem; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }
.bw-coord-value { font-size: .95rem; font-weight: 700; color: var(--drone-dark); margin: 6px 0 2px; }

/* Summary metric cards */
.bw-metric {
    background: white; border: 1px solid var(--drone-lblue);
    border-radius: 20px; padding: 1.1rem 1.25rem;
    display: flex; align-items: center; gap: 14px; margin-bottom: .75rem;
}
.bw-metric-icon {
    width: 44px; height: 44px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0;
}
.bw-metric-icon.neutral { background: var(--drone-bg); color: var(--drone-dark); }
.bw-metric-icon.fishing { background: #fee2e2; color: #dc2626; }
.bw-metric-icon.nonfishing { background: #d1fae5; color: #059669; }
.bw-metric-title { font-size: .7rem; font-weight: 700; color: #6b7280; text-transform: uppercase; }
.bw-metric-value { font-size: 1.6rem; font-weight: 900; color: var(--drone-dark); }
.bw-metric-value.fishing { color: #991b1b; }
.bw-metric-value.nonfishing { color: #065f46; }

/* Badge prediksi di tabel */
.badge-fishing {
    background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;
    padding: 2px 10px; border-radius: 999px; font-size: .68rem; font-weight: 700;
}
.badge-nonfishing {
    background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0;
    padding: 2px 10px; border-radius: 999px; font-size: .68rem; font-weight: 700;
}

/* Error banner */
.bw-error {
    background: #fff1f2; border-left: 4px solid #ef4444;
    border-radius: 12px; padding: 14px 16px;
    display: flex; gap: 10px; align-items: flex-start;
    margin-bottom: 1rem;
}
.bw-error-icon { color: #ef4444; font-size: 1rem; margin-top: 2px; }
.bw-error-title { font-size: .82rem; font-weight: 700; color: #7f1d1d; }
.bw-error-msg { font-size: .75rem; color: #b91c1c; margin-top: 2px; }

/* Guide table */
.bw-guide-table { width: 100%; border-collapse: collapse; font-size: .78rem; }
.bw-guide-table th {
    background: var(--drone-bg); color: var(--drone-dark);
    font-weight: 700; padding: 10px 12px; text-align: left;
    border-bottom: 1px solid var(--drone-lblue);
}
.bw-guide-table td { padding: 9px 12px; border-bottom: 1px solid rgba(203,227,239,.5); color: #374151; }
.bw-guide-table tr:last-child td { border-bottom: none; }
.bw-col-name { font-family: monospace; font-weight: 700; color: var(--drone-blue); }
.bw-guide-wrapper { border: 1px solid var(--drone-lblue); border-radius: 14px; overflow: hidden; }

/* Legend peta */
.bw-legend {
    display: flex; align-items: center; gap: 20px;
    background: rgba(237,243,251,.8); border: 1px solid var(--drone-lblue);
    border-radius: 12px; padding: 6px 14px; width: fit-content;
    font-size: .72rem; font-weight: 700;
}
.bw-legend-dot { width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; flex-shrink: 0; }

/* Dropzone */
.bw-dropzone {
    border: 2px dashed var(--drone-lblue); border-radius: 18px;
    background: rgba(237,243,251,.3); padding: 2.5rem 1rem;
    text-align: center; transition: all .2s; cursor: pointer;
}
.bw-dropzone:hover { border-color: var(--drone-blue); background: rgba(237,243,251,.7); }
.bw-dropzone-icon { font-size: 2.5rem; color: rgba(90,168,214,.6); margin-bottom: 10px; }
.bw-dropzone-title { font-size: .85rem; font-weight: 700; color: var(--drone-dark); }
.bw-dropzone-sub { font-size: .72rem; color: #9ca3af; margin-top: 4px; }

/* Footer */
.bw-footer { background: var(--drone-dark); color: rgba(255,255,255,.7); padding: 3rem 2rem 2rem; margin-top: 3rem; }
.bw-footer-brand { font-size: 1.1rem; font-weight: 800; color: white; }
.bw-footer-brand span { color: var(--drone-blue); }
.bw-footer-desc { font-size: .75rem; color: rgba(255,255,255,.5); line-height: 1.6; max-width: 340px; margin-top: 10px; }
.bw-footer-h { font-size: .7rem; font-weight: 700; color: white; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 10px; }
.bw-footer-item { font-size: .75rem; color: rgba(255,255,255,.45); margin-bottom: 6px; }
.bw-footer-copy { font-size: .72rem; color: rgba(255,255,255,.3); border-top: 1px solid rgba(255,255,255,.08); padding-top: 1.5rem; margin-top: 2rem; }

/* Force streamlit elements to blend */
.stButton > button {
    background: var(--drone-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 12px 24px !important;
    transition: all .2s !important;
    box-shadow: 0 4px 12px rgba(90,168,214,.3) !important;
}
.stButton > button:hover { background: rgba(90,168,214,.85) !important; }

.stDownloadButton > button {
    background: #059669 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 8px 18px !important;
    font-size: .8rem !important;
}

[data-testid="stFileUploader"] {
    background: rgba(237,243,251,.3);
    border: 2px dashed var(--drone-lblue);
    border-radius: 18px;
    padding: 1rem;
}

div[data-baseweb="select"] > div { border-radius: 12px !important; border-color: var(--drone-lblue) !important; }
div[data-baseweb="input"] > div { border-radius: 12px !important; border-color: var(--drone-lblue) !important; }
.stSlider > div { padding: 0 4px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & ARTEFAK
# ============================================================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_DIR / "model_cnn_laut_banda.keras")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    flag_enc = joblib.load(MODEL_DIR / "flag_encoder.pkl")
    gear_enc = joblib.load(MODEL_DIR / "gear_encoder.pkl")
    return model, scaler, flag_enc, gear_enc

model, scaler, flag_enc, gear_enc = load_artifacts()

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    def safe_encode(encoder, series, col_name):
        known = set(encoder.classes_)
        bad = ~series.isin(known)
        if bad.any():
            raise ValueError(
                f"Nilai tidak dikenal pada kolom '{col_name}': {series[bad].unique().tolist()}. "
                f"Nilai valid: {list(encoder.classes_)}"
            )
        return encoder.transform(series)

    work["flag_enc"]     = safe_encode(flag_enc, work["flag"].astype(str), "flag")
    work["geartype_enc"] = safe_encode(gear_enc, work["geartype"].astype(str), "geartype")

    feat = pd.DataFrame({
        "month":        work["month"],
        "day":          work["day"],
        "weekday":      work["weekday"],
        "cell_ll_lat":  work["cell_ll_lat"],
        "cell_ll_lon":  work["cell_ll_lon"],
        "flag":         work["flag_enc"],
        "geartype":     work["geartype_enc"],
        "hours":        work["hours"],
        "mmsi_present": work["mmsi_present"],
    })[FEATURE_ORDER]

    X = scaler.transform(feat).reshape(-1, 9, 1)
    probs = model.predict(X, verbose=0).flatten()
    result = df.copy()
    result["probabilitas_fishing"] = probs
    result["prediksi"] = np.where(probs >= 0.5, "Fishing", "Non-Fishing")
    return result

def make_map(points: list, center_lat=-5.5, center_lon=127.5, zoom=6) -> folium.Map:
    """points: list of dict {lat, lon, label, prob, geartype, flag}"""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
        scrollWheelZoom=False,
    )
    for p in points:
        color = "#f43f5e" if p["label"] == "Fishing" else "#3b82f6"
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=9,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            popup=folium.Popup(
                f"<b style='font-size:13px'>{p['label'].upper()}</b><br>"
                f"Probabilitas: {p['prob']*100:.1f}%<br>"
                f"Koordinat: {p['lat']:.4f}, {p['lon']:.4f}<br>"
                f"Alat tangkap: {p.get('geartype','')}<br>"
                f"Bendera: {p.get('flag','')}",
                max_width=200
            ),
            tooltip=f"{p['label']} ({p['prob']*100:.1f}%)"
        ).add_to(m)
    return m

# ============================================================
# PROMO BANNER
# ============================================================
st.markdown("""
<div class="bw-promo">
    Sistem Monitoring Maritim Integratif Laut Banda &nbsp;&nbsp;
</div>
""", unsafe_allow_html=True)

# ============================================================
# NAVBAR
# ============================================================
with open("app/static/logo.svg", "rb") as f:
    svg = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div class="bw-nav">
    <div style="display:flex;align-items:center;gap:12px">
        <img src="data:image/svg+xml;base64,{svg}" width="28">
        <div>
            <div class="bw-brand">MATA<span>BANDA</span></div>
            <div class="bw-subbrand">CNN Fish Predictor</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "manual"

# ============================================================
# LAYOUT UTAMA
# ============================================================
st.markdown('<div class="bw-page">', unsafe_allow_html=True)
left_col, right_col = st.columns([4, 8], gap="large")

# ================================================================
# SIDEBAR KIRI — Info & Arsitektur
# ================================================================
with left_col:
    # Keterangan penelitian
    st.markdown("""
    <div class="bw-card">
        <div class="bw-card-title"><span class="bw-icon"></span> Keterangan Penelitian</div>
        <p style="font-size:.82rem;color:#4b5563;line-height:1.65;margin-bottom:14px">
            Sistem ini merupakan implementasi model klasifikasi berbasis
            <strong>Convolutional Neural Network (CNN)</strong> yang dirancang untuk
            mengidentifikasi apakah sebuah kapal sedang melakukan aktivitas penangkapan ikan
            (<strong>Fishing</strong>) atau sekadar berlayar (<strong>Non-Fishing</strong>)
            di perairan Laut Banda berdasarkan data AIS dari <em>Global Fishing Watch</em>.
        </p>
        <div style="background:var(--drone-bg);border:1px solid var(--drone-lblue);border-radius:14px;padding:12px">
            <div style="font-size:.68rem;font-weight:700;color:var(--drone-dark);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">
                Wilayah Cakupan Model
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:.78rem;color:#4b5563">
                <div><span style="display:block;font-weight:700;color:var(--drone-dark)">Latitude</span>-8.0° s.d. -3.0°</div>
                <div><span style="display:block;font-weight:700;color:var(--drone-dark)">Longitude</span>123.0° s.d. 132.0°</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Arsitektur CNN
    st.markdown("""
    <div class="bw-card">
        <div class="bw-card-title"><span class="bw-icon"></span> Arsitektur CNN Model</div>
        <div class="bw-timeline">
            <div class="bw-tl-item">
                <div class="bw-tl-dot"></div>
                <div class="bw-tl-title">Input Layer (9 Fitur)</div>
                <div class="bw-tl-desc">Koordinat spasial, penanda waktu temporal, MMSI, serta representasi numerik encoder bendera dan alat tangkap.</div>
            </div>
            <div class="bw-tl-item">
                <div class="bw-tl-dot"></div>
                <div class="bw-tl-title">Conv1D(32) → MaxPool1D</div>
                <div class="bw-tl-desc">Mengekstrak pola lokal dari urutan 9 fitur yang telah dinormalisasi.</div>
            </div>
            <div class="bw-tl-item">
                <div class="bw-tl-dot"></div>
                <div class="bw-tl-title">Conv1D(64) → Flatten</div>
                <div class="bw-tl-desc">Ekstraksi fitur lebih dalam sebelum diratakan menjadi vektor 1D.</div>
            </div>
            <div class="bw-tl-item">
                <div class="bw-tl-dot dark"></div>
                <div class="bw-tl-title">Dense(64) → Dropout → Dense(1, sigmoid)</div>
                <div class="bw-tl-desc">Klasifikasi biner final — probabilitas Fishing vs Non-Fishing (threshold 0.5).</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Referensi label encoder
    flag_chips   = "".join(f'<span class="bw-chip">{k}</span>' for k in VALID_FLAGS)
    gear_chips   = "".join(f'<span class="bw-chip">{k}</span>' for k in VALID_GEARS)
    st.markdown(f"""
    <div class="bw-card">
        <div class="bw-card-title"><span class="bw-icon"></span> Referensi Label Encoder</div>
        <div style="margin-bottom:10px">
            <div style="font-size:.75rem;font-weight:700;color:var(--drone-dark);margin-bottom:5px">Negara Bendera (flag):</div>
            <div>{flag_chips}</div>
        </div>
        <div>
            <div style="font-size:.75rem;font-weight:700;color:var(--drone-dark);margin-bottom:5px">Alat Tangkap (geartype):</div>
            <div>{gear_chips}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# MAIN CONTENT KANAN
# ================================================================
with right_col:

    # Hero Section
    st.markdown("""
    <div class="bw-hero">
        <div class="bw-ship-bg"></div>
        <div class="bw-hero-badge">✦ Continuous Monitoring Platform</div>
        <h1>Deteksi & Klasifikasi Aktivitas Penangkapan Ikan</h1>
        <p>Masukkan koordinat aktivitas kapal secara manual atau unggah kumpulan data armada
        dalam format CSV untuk memprediksi pola penangkapan ikan di Laut Banda secara langsung.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tab Selector
    tab_col1, tab_col2, tab_col3 = st.columns([2, 2, 8])
    with tab_col1:
        if st.button("  Input Manual", use_container_width=True,
                     type="primary" if st.session_state.active_tab == "manual" else "secondary"):
            st.session_state.active_tab = "manual"
            st.rerun()
    with tab_col2:
        if st.button("  Upload CSV", use_container_width=True,
                     type="primary" if st.session_state.active_tab == "csv" else "secondary"):
            st.session_state.active_tab = "csv"
            st.rerun()

    # ── TAB MANUAL ──────────────────────────────────────────────
    if st.session_state.active_tab == "manual":

        st.markdown("""
        <div class="bw-card" style="margin-top:1rem">
            <div class="bw-card-title">
                <span class="bw-icon"></span> Konfigurasi Fitur Spasio-Temporal
                <span style="margin-left:auto;font-size:.7rem;font-family:monospace;color:#9ca3af">day = 1 (Konstan)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            month = st.slider(" Month — Bulan aktivitas (1–12)", 1, 12, 6,
                help="Bulan saat aktivitas kapal terjadi. Model menangkap pola musiman, "
                     "misal musim angin timur (Juni–Oktober) berbeda dengan musim barat.")
            weekday = st.selectbox(" Weekday — Hari dalam seminggu",
                options=list(range(7)),
                format_func=lambda x: f"{['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][x]} ({x})",
                index=1,
                help="0 = Senin … 6 = Minggu. Beberapa armada punya pola hari operasi yang khas.")
            hours = st.number_input(" Hours — Jam aktivitas terdeteksi",
                min_value=0.0, max_value=24.0, value=4.0, step=0.5,
                help="Total durasi (jam) kapal terdeteksi di sel grid ini dalam satu hari. "
                     "Semakin lama menetap di satu titik → semakin besar indikasi fishing.")

        with c2:
            lat = st.slider(" Latitude (cell_ll_lat)", LAT_MIN, LAT_MAX, -5.5, step=0.05,
                help="Lintang sudut kiri-bawah sel grid AIS. Nilai negatif = selatan khatulistiwa.")
            lon = st.slider(" Longitude (cell_ll_lon)", LON_MIN, LON_MAX, 127.5, step=0.05,
                help="Bujur sudut kiri-bawah sel grid AIS. Wilayah Laut Banda: 123°–132°BT.")
            mmsi_present = st.number_input(" MMSI Present — Jumlah kapal unik",
                min_value=1.0, max_value=50.0, value=1.0, step=1.0,
                help="Jumlah kapal berbeda (berdasarkan nomor MMSI) di sel grid ini. "
                     ">1 berarti beberapa kapal beroperasi berdekatan.")

        c3, c4 = st.columns(2)
        with c3:
            flag_keys = list(FLAGS_LABEL.keys())
            flag = st.selectbox("🏴 Flag — Bendera kapal",
                options=flag_keys,
                format_func=lambda x: FLAGS_LABEL[x],
                index=flag_keys.index("IDN") if "IDN" in flag_keys else 0,
                help="Negara bendera (flag state) tempat kapal terdaftar secara hukum.")
        with c4:
            gear_keys = list(GEAR_LABEL.keys())
            geartype = st.selectbox(" Geartype — Alat tangkap",
                options=gear_keys,
                format_func=lambda x: GEAR_LABEL[x],
                help="Jenis alat tangkap yang diidentifikasi GFW berdasarkan pola gerak kapal.")

        predict_btn = st.button("  Prediksi Aktivitas Kapal", use_container_width=True, type="primary")

        if predict_btn:
            input_df = pd.DataFrame([{
                "month": month, "day": 1, "weekday": weekday,
                "cell_ll_lat": lat, "cell_ll_lon": lon,
                "flag": flag, "geartype": geartype,
                "hours": hours, "mmsi_present": mmsi_present,
            }])
            try:
                result = predict_batch(input_df)
                pred_label = result["prediksi"].iloc[0]
                prob = float(result["probabilitas_fishing"].iloc[0])

                st.markdown('<div class="bw-card" style="margin-top:1rem">', unsafe_allow_html=True)
                st.markdown('<div class="bw-card-title"><span class="bw-icon"></span> Hasil Prediksi Model</div>', unsafe_allow_html=True)

                r1, r2, r3 = st.columns(3)
                with r1:
                    cls = "fishing" if pred_label == "Fishing" else "nonfishing"
                    desc = ("Sistem mengklasifikasikan pola trajektori sebagai aktivitas penangkapan ikan."
                            if pred_label == "Fishing"
                            else "Sistem menyimpulkan kapal sedang dalam status berlayar bebas (transit).")
                    st.markdown(f"""
                    <div class="bw-result-{cls}">
                        <div class="bw-prob-label">Kategori Deteksi</div>
                        <div class="bw-result-label">{pred_label.upper()}</div>
                        <div class="bw-result-desc">{desc}</div>
                    </div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div class="bw-prob-card">
                        <div class="bw-prob-label">Probabilitas Kecocokan</div>
                        <div class="bw-prob-value">{prob*100:.1f}%</div>
                        <div class="bw-progress-bar">
                            <div class="bw-progress-fill" style="width:{prob*100:.1f}%"></div>
                        </div>
                        <div style="font-size:.65rem;color:#9ca3af;margin-top:4px">Threshold klasifikasi: 50%</div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    st.markdown(f"""
                    <div class="bw-coord-card">
                        <div class="bw-coord-label">Target Spasial</div>
                        <div class="bw-coord-value">Lat: {lat:.4f}, Lon: {lon:.4f}</div>
                        <div style="font-size:.72rem;color:#6b7280">Sektor Laut Banda</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Peta manual
                st.markdown('<div class="bw-card">', unsafe_allow_html=True)
                st.markdown("""
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
                    <div class="bw-card-title" style="margin-bottom:0"> Visualisasi Spasio-Temporal</div>
                    <div class="bw-legend">
                        <div style="display:flex;align-items:center;gap:5px">
                            <div class="bw-legend-dot" style="background:#f43f5e"></div>
                            <span style="color:#9f1239">Fishing</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:5px">
                            <div class="bw-legend-dot" style="background:#3b82f6"></div>
                            <span style="color:#1e40af">Non-Fishing</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                m = make_map([{"lat": lat, "lon": lon, "label": pred_label,
                               "prob": prob, "geartype": geartype, "flag": flag}],
                             center_lat=lat, center_lon=lon, zoom=7)
                st_folium(m, height=380, use_container_width=True, returned_objects=[])
                st.markdown("</div>", unsafe_allow_html=True)

            except ValueError as e:
                st.markdown(f"""
                <div class="bw-error">
                    <div class="bw-error-icon"></div>
                    <div>
                        <div class="bw-error-title">Gagal Memvalidasi Data</div>
                        <div class="bw-error-msg">{e}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── TAB CSV ─────────────────────────────────────────────────
    else:
        st.markdown('<div style="margin-top:1rem">', unsafe_allow_html=True)

        # Dropzone upload
        st.markdown("""
        <div class="bw-card">
            <div class="bw-card-title"><span class="bw-icon"></span> Pengunggahan Dataset CSV Spasio-Temporal</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Seret & letakkan file CSV di sini, atau klik untuk memilih",
            type=["csv"], label_visibility="visible"
        )

        # Panduan kolom
        st.markdown("""
        <div class="bw-card" style="margin-top:.75rem">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
                <div class="bw-card-title" style="margin-bottom:0"> Tabel Panduan Struktur Kolom</div>
            </div>
            <div class="bw-guide-wrapper">
            <table class="bw-guide-table">
                <thead><tr><th>Nama Kolom</th><th>Tipe</th><th>Deskripsi & Ketentuan</th></tr></thead>
                <tbody>
                    <tr><td class="bw-col-name">month</td><td>Integer</td><td>Bulan aktivitas (1–12)</td></tr>
                    <tr><td class="bw-col-name">day</td><td>Integer</td><td>Tanggal — isi <code>1</code> (konstan pada data training)</td></tr>
                    <tr><td class="bw-col-name">weekday</td><td>Integer</td><td>Hari dalam seminggu (0=Senin … 6=Minggu)</td></tr>
                    <tr><td class="bw-col-name">cell_ll_lat</td><td>Float</td><td>Latitude sel grid AIS (−8.0 s.d. −3.0)</td></tr>
                    <tr><td class="bw-col-name">cell_ll_lon</td><td>Float</td><td>Longitude sel grid AIS (123.0 s.d. 132.0)</td></tr>
                    <tr><td class="bw-col-name">flag</td><td>String</td><td>Kode bendera kapal, contoh: <code>IDN</code>, <code>CHN</code></td></tr>
                    <tr><td class="bw-col-name">geartype</td><td>String</td><td>Jenis alat tangkap, contoh: <code>fishing</code>, <code>trawlers</code></td></tr>
                    <tr><td class="bw-col-name">hours</td><td>Float</td><td>Total jam aktivitas terdeteksi di sel grid (0–24)</td></tr>
                    <tr><td class="bw-col-name">mmsi_present</td><td>Integer</td><td>Jumlah kapal unik (MMSI) terdeteksi di sel grid</td></tr>
                </tbody>
            </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download template
        sample_csv = pd.DataFrame([
            {"month":6,"day":1,"weekday":2,"cell_ll_lat":-5.5,"cell_ll_lon":127.5,
             "flag":"IDN","geartype":"fishing","hours":5.0,"mmsi_present":1},
            {"month":8,"day":1,"weekday":4,"cell_ll_lat":-6.2,"cell_ll_lon":129.1,
             "flag":"CHN","geartype":"purse_seines","hours":12.0,"mmsi_present":2},
            {"month":10,"day":1,"weekday":0,"cell_ll_lat":-7.0,"cell_ll_lon":130.3,
             "flag":"JPN","geartype":"trawlers","hours":8.5,"mmsi_present":3},
        ])
        st.download_button(" Unduh Template CSV",
            data=sample_csv.to_csv(index=False).encode("utf-8"),
            file_name="template_matabanda.csv", mime="text/csv")

        # Proses file yang diupload
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                missing_cols = [c for c in FEATURE_ORDER if c not in raw_df.columns]
                if missing_cols:
                    st.markdown(f"""
                    <div class="bw-error">
                        <div class="bw-error-icon"></div>
                        <div>
                            <div class="bw-error-title">Struktur CSV Salah</div>
                            <div class="bw-error-msg">Kolom wajib berikut tidak ditemukan: {missing_cols}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    with st.spinner("Memproses prediksi massal..."):
                        result = predict_batch(raw_df)

                    total      = len(result)
                    n_fishing  = int((result["prediksi"] == "Fishing").sum())
                    n_nonfishing = total - n_fishing

                    # Ringkasan metrik
                    st.markdown("""<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:1rem 0">""", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""
                        <div class="bw-metric">
                            <div class="bw-metric-icon neutral"></div>
                            <div>
                                <div class="bw-metric-title">Total Titik Data</div>
                                <div class="bw-metric-value">{total}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="bw-metric">
                            <div class="bw-metric-icon fishing"></div>
                            <div>
                                <div class="bw-metric-title">Fishing Terdeteksi</div>
                                <div class="bw-metric-value fishing">{n_fishing}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="bw-metric">
                            <div class="bw-metric-icon nonfishing"></div>
                            <div>
                                <div class="bw-metric-title">Non-Fishing</div>
                                <div class="bw-metric-value nonfishing">{n_nonfishing}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    # Peta massal
                    st.markdown('<div class="bw-card">', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
                        <div class="bw-card-title" style="margin-bottom:0"> Peta Visualisasi Spasio-Temporal (Laut Banda)</div>
                        <div class="bw-legend">
                            <div style="display:flex;align-items:center;gap:5px">
                                <div class="bw-legend-dot" style="background:#f43f5e"></div>
                                <span style="color:#9f1239">Fishing</span>
                            </div>
                            <div style="display:flex;align-items:center;gap:5px">
                                <div class="bw-legend-dot" style="background:#3b82f6"></div>
                                <span style="color:#1e40af">Non-Fishing</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    points = [
                        {
                            "lat": row["cell_ll_lat"], "lon": row["cell_ll_lon"],
                            "label": row["prediksi"],
                            "prob": float(row["probabilitas_fishing"]),
                            "geartype": row.get("geartype", ""),
                            "flag": row.get("flag", ""),
                        }
                        for _, row in result.iterrows()
                    ]
                    center_lat = float(result["cell_ll_lat"].mean())
                    center_lon = float(result["cell_ll_lon"].mean())
                    m = make_map(points, center_lat=center_lat, center_lon=center_lon, zoom=6)
                    st_folium(m, height=420, use_container_width=True, returned_objects=[])
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Tabel hasil detail
                    st.markdown('<div class="bw-card">', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
                        <div class="bw-card-title" style="margin-bottom:0"> Tabel Hasil Prediksi Detail</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Render tabel HTML dengan badge
                    rows_html = ""
                    for i, row in result.iterrows():
                        badge_cls = "badge-fishing" if row["prediksi"] == "Fishing" else "badge-nonfishing"
                        rows_html += f"""
                        <tr>
                            <td style="padding:9px 12px;font-weight:600;color:var(--drone-dark)">{i+1}</td>
                            <td style="padding:9px 12px"><span class="bw-chip">{row.get('flag','')}</span></td>
                            <td style="padding:9px 12px;color:#374151">{row.get('geartype','')}</td>
                            <td style="padding:9px 12px;text-align:center;font-family:monospace">{row['cell_ll_lat']:.4f}</td>
                            <td style="padding:9px 12px;text-align:center;font-family:monospace">{row['cell_ll_lon']:.4f}</td>
                            <td style="padding:9px 12px;text-align:right;font-family:monospace;font-weight:700;color:var(--drone-blue)">{row['probabilitas_fishing']*100:.1f}%</td>
                            <td style="padding:9px 12px;text-align:center"><span class="{badge_cls}">{row['prediksi'].upper()}</span></td>
                        </tr>"""

                    st.markdown(f"""
                    <div class="bw-guide-wrapper" style="max-height:380px;overflow-y:auto">
                    <table class="bw-guide-table">
                        <thead><tr>
                            <th>#</th><th>Flag</th><th>Alat Tangkap</th>
                            <th style="text-align:center">Latitude</th>
                            <th style="text-align:center">Longitude</th>
                            <th style="text-align:right">Probabilitas</th>
                            <th style="text-align:center">Hasil Prediksi</th>
                        </tr></thead>
                        <tbody style="divide-y">{rows_html}</tbody>
                    </table>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Download hasil
                    st.download_button(
                        " Unduh Hasil Prediksi (CSV)",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi_MataBanda.csv",
                        mime="text/csv",
                    )

            except ValueError as e:
                st.markdown(f"""
                <div class="bw-error">
                    <div class="bw-error-icon"></div>
                    <div>
                        <div class="bw-error-title">Gagal Memvalidasi Data</div>
                        <div class="bw-error-msg">{e}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="bw-error">
                    <div class="bw-error-icon"></div>
                    <div>
                        <div class="bw-error-title">Error Tidak Terduga</div>
                        <div class="bw-error-msg">{e}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # .bw-page

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="bw-footer">
    <div style="max-width:1280px;margin:0 auto;display:grid;grid-template-columns:2fr 1fr 1fr;gap:2rem">
        <div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                <div style="background:var(--drone-blue);color:white;padding:7px 9px;border-radius:10px">⚓</div>
                <div class="bw-footer-brand">MATA<span>BANDA</span></div>
            </div>
            <div class="bw-footer-desc">
                Platform spasio-temporal untuk pengawasan aktivitas maritim, pencegahan IUU Fishing,
                dan pelestarian ekosistem Laut Banda menggunakan Deep Learning CNN.
            </div>
        </div>
        <div>
            <div class="bw-footer-h">Teknologi</div>
            <div class="bw-footer-item">CNN 1D (TensorFlow/Keras)</div>
            <div class="bw-footer-item">LabelEncoder (scikit-learn)</div>
            <div class="bw-footer-item">Folium — Carto Basemap</div>
        </div>
        <div>
            <div class="bw-footer-h">Informasi Spasial</div>
            <div class="bw-footer-item">Lat: −8.0° s.d. −3.0° LS</div>
            <div class="bw-footer-item">Lon: 123.0° s.d. 132.0° BT</div>
            <div class="bw-footer-item">Wilayah: Laut Banda, Maluku</div>
        </div>
    </div>
    <div class="bw-footer-copy" style="max-width:1280px;margin:0 auto">
        &copy; 2026 MataBanda —  Tugas Machine Learning Kelompok 11.
    </div>
</div>
""", unsafe_allow_html=True)