import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import database as db
from modules.auth import require_login, auth_sidebar_footer
from modules import lexicon as lex
from modules.utils import inject_base_style, page_header, section_title, metric_card

st.set_page_config(page_title="Labeling Sentimen", page_icon="🏷️", layout="wide")
inject_base_style()
require_login()
db.init_db()
auth_sidebar_footer()

page_header(
    icon="🏷️",
    title="Pelabelan Sentimen (Lexicon-Based)",
    subtitle="Label ditentukan dari skor kata positif vs. negatif pada teks hasil preprocessing (final_text). "
             "Leksikon dapat dikustomisasi langsung dari halaman ini.",
    tags=["😊 Positif", "😠 Negatif", "📚 Leksikon Dapat Diedit"],
    gradient="orange",
)

if not db.is_seeded():
    st.warning("Database masih kosong. Muat dataset awal dari halaman Beranda terlebih dahulu.")
    st.stop()

# ------------------------------------------------------------------ #
# Session-state lexicon (dapat ditambah lewat UI, tanpa ubah file)
# ------------------------------------------------------------------ #
if "positive_words" not in st.session_state:
    st.session_state.positive_words = set(lex.POSITIVE_WORDS)
if "negative_words" not in st.session_state:
    st.session_state.negative_words = set(lex.NEGATIVE_WORDS)

tab_run, tab_lexicon, tab_result = st.tabs(
    ["⚙️ Jalankan Labeling", "📚 Kelola Leksikon", "📊 Hasil & Distribusi"]
)

# ============================================================ #
# TAB: RUN LABELING
# ============================================================ #
with tab_run:
    stats = db.get_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Komentar", stats["total"])
    c2.metric("Sudah Diproses (siap dilabeli)", stats["preprocessed"])
    c3.metric("Sudah Dilabeli", stats["labeled"])

    if stats["preprocessed"] == 0:
        st.warning("Belum ada data yang di-preprocessing. Jalankan halaman **🧹 Preprocessing** terlebih dahulu.")
    else:
        only_new = st.checkbox("Hanya labeli komentar yang belum berlabel", value=True)
        target_df = db.fetch_all(preprocessed_only=True)
        if only_new:
            target_df = target_df[target_df["label"].isna()]

        st.write(f"Akan melabeli **{len(target_df)}** komentar menggunakan leksikon saat ini "
                 f"({len(st.session_state.positive_words)} kata positif, "
                 f"{len(st.session_state.negative_words)} kata negatif).")

        if st.button("🚀 Jalankan Pelabelan", type="primary", disabled=(len(target_df) == 0)):
            progress = st.progress(0, text="Melabeli...")
            rows = []
            n = len(target_df)
            for i, r in enumerate(target_df.itertuples()):
                label = lex.label_from_text(
                    r.final_text or "",
                    st.session_state.positive_words,
                    st.session_state.negative_words,
                )
                rows.append((label, r.id))
                if (i + 1) % 50 == 0 or (i + 1) == n:
                    progress.progress((i + 1) / n, text=f"Melabeli {i+1}/{n}")
            db.bulk_update_labels(rows)
            st.success(f"✅ {n} komentar berhasil dilabeli.")
            st.rerun()

# ============================================================ #
# TAB: MANAGE LEXICON (CRUD kecil untuk daftar kata)
# ============================================================ #
with tab_lexicon:
    section_title("📚", "Kelola Daftar Kata Leksikon")
    st.caption("Perubahan berlaku untuk sesi ini dan akan dipakai saat tombol 'Jalankan Pelabelan' ditekan.")

    colp, coln = st.columns(2)
    with colp:
        st.markdown("#### 😊 Kata Positif")
        new_pos = st.text_input("Tambah kata positif baru", key="add_pos")
        if st.button("➕ Tambah ke Positif") and new_pos.strip():
            st.session_state.positive_words.add(new_pos.strip().lower())
            st.session_state.negative_words.discard(new_pos.strip().lower())
            st.rerun()

        for w in sorted(st.session_state.positive_words):
            c1, c2 = st.columns([4, 1])
            c1.write(f"🟢 {w}")
            if c2.button("🗑️", key=f"del_pos_{w}"):
                st.session_state.positive_words.discard(w)
                st.rerun()

    with coln:
        st.markdown("#### 😠 Kata Negatif")
        new_neg = st.text_input("Tambah kata negatif baru", key="add_neg")
        if st.button("➕ Tambah ke Negatif") and new_neg.strip():
            st.session_state.negative_words.add(new_neg.strip().lower())
            st.session_state.positive_words.discard(new_neg.strip().lower())
            st.rerun()

        for w in sorted(st.session_state.negative_words):
            c1, c2 = st.columns([4, 1])
            c1.write(f"🔴 {w}")
            if c2.button("🗑️", key=f"del_neg_{w}"):
                st.session_state.negative_words.discard(w)
                st.rerun()

    st.divider()
    if st.button("↩️ Kembalikan ke Leksikon Default (notebook asli)"):
        st.session_state.positive_words = set(lex.POSITIVE_WORDS)
        st.session_state.negative_words = set(lex.NEGATIVE_WORDS)
        st.rerun()

# ============================================================ #
# TAB: RESULTS
# ============================================================ #
with tab_result:
    stats = db.get_stats()
    if stats["labeled"] == 0:
        st.info("Belum ada data berlabel. Jalankan pelabelan terlebih dahulu.")
    else:
        df = db.fetch_all()
        labeled = df[df["label"].notna()].copy()
        labeled["Sentimen"] = labeled["label"].map({1: "Positif", 0: "Negatif"})

        c1, c2 = st.columns(2)
        with c1:
            st.caption("📖 Jumlah komentar berdasarkan hasil label (angka absolut).")
            counts = labeled["Sentimen"].value_counts().reset_index()
            counts.columns = ["Sentimen", "Jumlah"]
            fig = px.bar(counts, x="Sentimen", y="Jumlah", color="Sentimen", text="Jumlah",
                         color_discrete_map={"Positif": "#22c55e", "Negatif": "#ef4444"})
            fig.update_layout(showlegend=False, height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.caption("📖 Proporsi (%) sentimen positif vs. negatif dari seluruh data berlabel.")
            pct = (labeled["Sentimen"].value_counts(normalize=True) * 100).round(2).reset_index()
            pct.columns = ["Sentimen", "Persentase"]
            fig2 = px.pie(pct, names="Sentimen", values="Persentase", hole=0.45,
                          color="Sentimen",
                          color_discrete_map={"Positif": "#22c55e", "Negatif": "#ef4444"})
            fig2.update_layout(height=360)
            st.plotly_chart(fig2, use_container_width=True)

        section_title("📱", "Distribusi Sentimen per Aplikasi AI")
        st.caption("📖 Bandingkan proporsi sentimen positif/negatif antar aplikasi AI untuk melihat mana yang "
                   "menerima respons paling positif dari pengguna.")
        cross = pd.crosstab(labeled["aplikasi"], labeled["Sentimen"])
        st.bar_chart(cross)

        section_title("📋", "Contoh Data Berlabel")
        f_sent = st.radio("Filter", ["Semua", "Positif", "Negatif"], horizontal=True)
        show = labeled if f_sent == "Semua" else labeled[labeled["Sentimen"] == f_sent]
        st.dataframe(
            show[["aplikasi", "komentar_asli", "final_text", "Sentimen"]].sample(min(15, len(show))),
            use_container_width=True, hide_index=True,
        )
