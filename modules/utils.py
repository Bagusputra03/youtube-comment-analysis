"""
================================================================================
MODULE: utils.py
Fungsi bantu bersama:
- Sistem desain (design system) terpusat: tema global, hero header per halaman,
  kartu statistik, badge, dan pembatas seksi — dipakai di seluruh halaman agar
  tampilan aplikasi konsisten, modern, dan informatif.
- Penyimpanan/pemuatan model terlatih (TF-IDF + Naive Bayes + SVM) ke disk.
================================================================================
"""

import os
import pickle
import datetime as dt

import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
NB_PATH = os.path.join(MODEL_DIR, "model_naive_bayes.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "model_svm.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.pkl")


# ================================================================== #
# DESIGN SYSTEM
# ================================================================== #
GRADIENTS = {
    "purple": "linear-gradient(135deg, #6a5cff 0%, #8e7bff 45%, #b39cff 100%)",
    "blue":   "linear-gradient(135deg, #2563eb 0%, #3b82f6 45%, #60a5fa 100%)",
    "green":  "linear-gradient(135deg, #059669 0%, #10b981 45%, #34d399 100%)",
    "orange": "linear-gradient(135deg, #ea580c 0%, #f97316 45%, #fb923c 100%)",
    "pink":   "linear-gradient(135deg, #db2777 0%, #ec4899 45%, #f472b6 100%)",
    "dark":   "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)",
    "red":    "linear-gradient(135deg, #b91c1c 0%, #ef4444 45%, #f87171 100%)",
}


def inject_base_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        h1, h2, h3, .hero-title { font-family: 'Poppins', sans-serif; }

        .main > div {padding-top: 1rem;}
        footer {visibility: hidden;}

        /* ---------- HERO HEADER ---------- */
        .hero-banner {
            background: var(--hero-gradient, linear-gradient(135deg,#6a5cff,#8e7bff));
            border-radius: 20px;
            padding: 30px 34px;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px rgba(80,60,200,0.25);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::after {
            content: "";
            position: absolute; top: -60px; right: -60px;
            width: 220px; height: 220px; border-radius: 50%;
            background: rgba(255,255,255,0.08);
        }
        .hero-banner::before {
            content: "";
            position: absolute; bottom: -80px; left: 30%;
            width: 260px; height: 260px; border-radius: 50%;
            background: rgba(255,255,255,0.06);
        }
        .hero-icon {
            font-size: 2.4rem; line-height:1;
            background: rgba(255,255,255,0.18);
            display:inline-flex; align-items:center; justify-content:center;
            width: 62px; height: 62px; border-radius: 16px;
            margin-bottom: 10px;
        }
        .hero-title {
            color: white; font-size: 1.9rem; font-weight: 700; margin: 0;
            position: relative; z-index: 1;
        }
        .hero-subtitle {
            color: rgba(255,255,255,0.92); font-size: 0.98rem; margin-top: 6px;
            position: relative; z-index: 1; max-width: 780px;
        }
        .hero-tags { margin-top: 14px; position: relative; z-index: 1; }
        .hero-tag {
            display:inline-block; background: rgba(255,255,255,0.16);
            color: white; font-size: 0.76rem; font-weight: 600;
            padding: 4px 12px; border-radius: 999px; margin-right: 8px;
            margin-bottom: 6px; backdrop-filter: blur(4px);
        }

        /* ---------- STAT / METRIC CARDS ---------- */
        .stat-card {
            background: var(--card-gradient, linear-gradient(135deg, #6a5cff 0%, #8e7bff 100%));
            padding: 18px 20px; border-radius: 16px; color: white;
            box-shadow: 0 6px 18px rgba(90,70,220,0.22);
            transition: transform 0.15s ease;
            height: 100%;
        }
        .stat-card:hover { transform: translateY(-3px); }
        .stat-card .stat-icon { font-size: 1.4rem; opacity: 0.95; }
        .stat-card .stat-value { margin: 4px 0 0 0; font-size: 1.85rem; font-weight: 700; }
        .stat-card .stat-label { margin: 0; opacity: 0.92; font-size: 0.8rem; font-weight: 500; }
        .stat-card .stat-sub { margin-top: 6px; font-size: 0.72rem; opacity: 0.85; }

        /* ---------- BADGES / PILLS ---------- */
        .pill {
            display:inline-block; padding: 4px 13px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; margin-right: 6px;
        }
        .pill-pos {background:#d7f8e8; color:#0a7a3d;}
        .pill-neg {background:#fde2e2; color:#b3261e;}
        .pill-muted {background:#eee; color:#666;}
        .pill-info {background:#dbeafe; color:#1d4ed8;}
        .pill-warn {background:#fef3c7; color:#92400e;}

        /* ---------- SECTION TITLE ---------- */
        .section-title {
            font-family:'Poppins',sans-serif; font-weight:700; font-size:1.15rem;
            display:flex; align-items:center; gap:8px; margin: 4px 0 10px 0;
            border-left: 5px solid #8e7bff; padding-left: 10px;
        }

        /* ---------- INFO / DANGER BOX ---------- */
        .info-box {
            background: #f4f2ff; border: 1px solid #e2ddff; border-radius: 12px;
            padding: 14px 18px; font-size: 0.9rem; color:#3b2f7a;
        }
        .danger-zone {
            background: repeating-linear-gradient(135deg, #fff5f5, #fff5f5 10px, #ffecec 10px, #ffecec 20px);
            border: 2px dashed #ef4444; border-radius: 16px; padding: 18px 20px;
            margin-top: 10px;
        }
        .danger-zone h4 { color:#b91c1c; margin:0 0 6px 0; }

        /* ---------- SIDEBAR ---------- */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(0,0,0,0.06);
        }
        section[data-testid="stSidebar"] .stButton button {
            border-radius: 10px;
        }

        /* ---------- BUTTONS ---------- */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6a5cff, #8e7bff) !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(106,92,255,0.35);
        }

        /* ---------- TABS ---------- */
        .stTabs [data-baseweb="tab"] {
            font-weight: 600; border-radius: 8px 8px 0 0;
        }

        /* ---------- DATAFRAME ---------- */
        [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

        /* ---------- MENYEMBUNYIKAN SIDEBAR BAWAAN STREAMLIT ---------- */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str = "", tags=None, gradient: str = "purple"):
    """Menampilkan hero-banner header yang konsisten untuk setiap halaman."""
    tags_html = ""
    if tags:
        tags_html = '<div class="hero-tags">' + "".join(
            f'<span class="hero-tag">{t}</span>' for t in tags
        ) + "</div>"

    st.markdown(f"""
        <div class="hero-banner" style="--hero-gradient: {GRADIENTS.get(gradient, GRADIENTS['purple'])};">
            <div class="hero-icon">{icon}</div>
            <p class="hero-title">{title}</p>
            <p class="hero-subtitle">{subtitle}</p>
            {tags_html}
        </div>
    """, unsafe_allow_html=True)


def section_title(icon: str, text: str):
    st.markdown(f'<div class="section-title">{icon} {text}</div>', unsafe_allow_html=True)


def metric_card(col, icon: str, label: str, value, sub: str = "", gradient: str = "purple"):
    """Merender satu kartu statistik ke dalam kolom Streamlit `col`."""
    with col:
        st.markdown(f"""
            <div class="stat-card" style="--card-gradient: {GRADIENTS.get(gradient, GRADIENTS['purple'])};">
                <div class="stat-icon">{icon}</div>
                <p class="stat-value">{value}</p>
                <p class="stat-label">{label}</p>
                {f'<p class="stat-sub">{sub}</p>' if sub else ''}
            </div>
        """, unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "muted"):
    return f'<span class="pill pill-{kind}">{text}</span>'


# ================================================================== #
# MODEL PERSISTENCE
# ================================================================== #
def save_models(vectorizer, nb_model, svm_model, meta: dict):
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(NB_PATH, "wb") as f:
        pickle.dump(nb_model, f)
    with open(SVM_PATH, "wb") as f:
        pickle.dump(svm_model, f)
    meta = {**meta, "saved_at": dt.datetime.now().isoformat(timespec="seconds")}
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)


def load_models():
    if not (os.path.exists(TFIDF_PATH) and os.path.exists(NB_PATH) and os.path.exists(SVM_PATH)):
        return None
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(NB_PATH, "rb") as f:
        nb_model = pickle.load(f)
    with open(SVM_PATH, "rb") as f:
        svm_model = pickle.load(f)
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    return {"vectorizer": vectorizer, "nb_model": nb_model, "svm_model": svm_model, "meta": meta}


def models_available() -> bool:
    return os.path.exists(TFIDF_PATH) and os.path.exists(NB_PATH) and os.path.exists(SVM_PATH)
