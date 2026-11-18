"""
================================================================================
APLIKASI: Analisis Sentimen Komentar YouTube tentang AI (ChatGPT, Gemini, dll)
Berbasis: Lexicon-Based Labeling + TF-IDF + Naive Bayes + SVM + LDA Topic Modeling
================================================================================
Halaman ini adalah Dashboard / Beranda. Gunakan menu di sidebar (folder
`pages/`) untuk berpindah ke modul CRUD, Preprocessing, Labeling, Pemodelan,
Topic Modeling, WordCloud, dan Prediksi.
================================================================================
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import database as db
from modules.utils import inject_base_style, models_available, page_header, metric_card, section_title, info_box
from modules.auth import require_login, auth_sidebar_footer

st.set_page_config(
    page_title="Analisis Sentimen Komentar YouTube AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_base_style()
require_login()
db.init_db()

auth_sidebar_footer()

# ------------------------------------------------------------------ #
# HEADER
# ------------------------------------------------------------------ #
page_header(
    icon="🤖",
    title="Dashboard Analisis Sentimen Komentar YouTube tentang AI",
    subtitle="Studi kasus komentar YouTube berbahasa Indonesia terkait aplikasi AI "
             "(ChatGPT, Gemini, Perplexity, dll.) — pipeline lengkap dari data mentah "
             "hingga model klasifikasi siap pakai.",
    tags=["🏷️ Lexicon Labeling", "🔠 TF-IDF", "🧠 Naive Bayes", "📐 SVM", "🧩 LDA Topic Modeling"],
    gradient="purple",
)

if not db.is_seeded():
    info_box("Muat dataset awal terlebih dahulu lewat sidebar untuk mulai menjelajah dashboard.")
    st.stop()

stats = db.get_stats()
progress_pct = round((stats["labeled"] / stats["total"]) * 100, 1) if stats["total"] else 0
pos_pct = round((stats["positif"] / stats["labeled"]) * 100, 1) if stats["labeled"] else 0

# ------------------------------------------------------------------ #
# KARTU STATISTIK
# ------------------------------------------------------------------ #
section_title("📌", "Ringkasan Data")
c1, c2, c3, c4, c5 = st.columns(5)
metric_card(c1, "🗂️", "Total Komentar", f"{stats['total']:,}", "seluruh data di database", "purple")
metric_card(c2, "🧹", "Sudah di-Preprocess", f"{stats['preprocessed']:,}",
            f"{round(stats['preprocessed']/stats['total']*100,1) if stats['total'] else 0}% dari total", "blue")
metric_card(c3, "🏷️", "Sudah Dilabeli", f"{stats['labeled']:,}", f"{progress_pct}% progres pelabelan", "orange")
metric_card(c4, "😊", "Sentimen Positif", f"{stats['positif']:,}", f"{pos_pct}% dari data berlabel", "green")
metric_card(c5, "😠", "Sentimen Negatif", f"{stats['negatif']:,}",
            f"{round(100-pos_pct,1) if stats['labeled'] else 0}% dari data berlabel", "red")

st.write("")
st.progress(min(progress_pct / 100, 1.0), text=f"Progres pelabelan dataset: {progress_pct}%")
st.write("")

if not models_available():
    info_box("💡 Model klasifikasi belum dilatih. Kunjungi halaman <b>🤖 Pemodelan & Evaluasi</b> "
              "setelah data dilabeli untuk melatih model Naive Bayes & SVM.")
else:
    st.markdown('<span class="pill pill-pos">✅ Model klasifikasi siap digunakan</span>', unsafe_allow_html=True)

info_box("📌 <b>Alur data:</b> Setiap kali data dimuat/diimpor (baik dari Excel maupun form Tambah Data), "
         "sistem <b>menyimpan teks apa adanya (raw)</b> sesuai dataset asli — <b>tidak ada preprocessing otomatis</b>. "
         "Kolom <i>clean_text</i>, <i>normalized_text</i>, <i>final_text</i>, dan <i>label</i> baru terisi setelah "
         "Anda menjalankan halaman <b>🧹 Preprocessing</b> dan <b>🏷️ Labeling Sentimen</b> secara manual.")

st.divider()

# ------------------------------------------------------------------ #
# VISUALISASI RINGKAS
# ------------------------------------------------------------------ #
df = db.fetch_all()

col_left, col_right = st.columns(2)

with col_left:
    section_title("📱", "Distribusi Komentar per Aplikasi AI")
    st.caption("📖 Grafik ini menunjukkan berapa banyak komentar yang berhasil dikumpulkan untuk masing-masing "
               "aplikasi AI (ChatGPT, Gemini, dll). Semakin tinggi batang, semakin banyak komentar yang tersedia "
               "untuk aplikasi tersebut.")
    app_counts = df["aplikasi"].value_counts().reset_index()
    app_counts.columns = ["Aplikasi", "Jumlah"]
    fig = px.bar(app_counts, x="Aplikasi", y="Jumlah", color="Aplikasi", text="Jumlah")
    fig.update_layout(showlegend=False, height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    section_title("😊😠", "Distribusi Sentimen")
    st.caption("📖 Diagram lingkaran ini memperlihatkan proporsi komentar bersentimen Positif vs. Negatif "
               "dari seluruh data yang **sudah dilabeli**. Berguna untuk melihat kecenderungan opini publik "
               "secara umum.")
    if stats["labeled"] > 0:
        labeled_df = df[df["label"].notna()].copy()
        labeled_df["Sentimen"] = labeled_df["label"].map({1: "Positif", 0: "Negatif"})
        sent_counts = labeled_df["Sentimen"].value_counts().reset_index()
        sent_counts.columns = ["Sentimen", "Jumlah"]
        fig2 = px.pie(sent_counts, names="Sentimen", values="Jumlah", hole=0.45,
                       color="Sentimen",
                       color_discrete_map={"Positif": "#22c55e", "Negatif": "#ef4444"})
        fig2.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        info_box("Belum ada data yang dilabeli. Kunjungi halaman <b>🏷️ Labeling Sentimen</b>.")

st.divider()

section_title("📊", "Aktivitas Komentar dari Waktu ke Waktu")
st.caption("📖 Grafik garis ini menunjukkan tren jumlah komentar yang masuk per tanggal, membantu melihat "
           "kapan diskusi tentang aplikasi AI sedang ramai.")
try:
    tmp = df.copy()
    tmp["tgl"] = pd.to_datetime(tmp["tanggal_komentar"], errors="coerce", utc=True).dt.date
    ts = tmp.dropna(subset=["tgl"]).groupby("tgl").size().reset_index(name="Jumlah Komentar")
    fig3 = px.line(ts, x="tgl", y="Jumlah Komentar", markers=True)
    fig3.update_layout(height=320, xaxis_title="Tanggal", yaxis_title="Jumlah Komentar",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig3.update_traces(line_color="#6a5cff", marker_color="#6a5cff")
    st.plotly_chart(fig3, use_container_width=True)
except Exception:
    st.caption("Tidak dapat membuat grafik deret waktu untuk data saat ini.")

st.divider()
section_title("🗂️", "Cuplikan Data Terbaru")
show_recent = df[["aplikasi", "author", "komentar_asli", "like_count", "label"]].head(10).copy()
show_recent["label"] = show_recent["label"].map({1: "😊 Positif", 0: "😠 Negatif"}).fillna("— Belum dilabeli")
show_recent.columns = ["Aplikasi", "Penulis", "Komentar", "Suka", "Sentimen"]
st.dataframe(show_recent, use_container_width=True, hide_index=True)

st.caption("💡 Gunakan menu di sidebar untuk mengelola data (CRUD), menjalankan preprocessing, "
           "pelabelan, pemodelan, topic modeling, wordcloud, hingga prediksi komentar baru.")
