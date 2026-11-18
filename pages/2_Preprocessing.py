import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from modules import database as db
from modules.auth import require_login, auth_sidebar_footer
from modules import preprocessing as pp
from modules.utils import inject_base_style, page_header, section_title, metric_card

st.set_page_config(page_title="Preprocessing Teks", page_icon="🧹", layout="wide")
inject_base_style()
require_login()
db.init_db()
auth_sidebar_footer()

page_header(
    icon="🧹",
    title="Preprocessing Teks Bahasa Indonesia",
    subtitle="Pipeline: Cleaning → Normalisasi Slang → Tokenisasi → Stopword Removal → Stemming (Sastrawi)",
    tags=["🧼 Cleaning", "📖 Normalisasi Slang", "✂️ Tokenisasi", "🚫 Stopword Removal", "🌱 Stemming"],
    gradient="green",
)

if not db.is_seeded():
    st.warning("Database masih kosong. Muat dataset awal dari halaman Beranda terlebih dahulu.")
    st.stop()

_stats = db.get_stats()
_c1, _c2, _c3 = st.columns(3)
metric_card(_c1, "🗂️", "Total Komentar", f"{_stats['total']:,}", gradient="green")
metric_card(_c2, "🧹", "Sudah Diproses", f"{_stats['preprocessed']:,}", gradient="blue")
metric_card(_c3, "⏳", "Belum Diproses", f"{_stats['total']-_stats['preprocessed']:,}", gradient="orange")
st.write("")

tab_tester, tab_bulk, tab_dict = st.tabs(
    ["🧪 Coba Interaktif", "⚙️ Jalankan pada Seluruh Data", "📖 Kamus Slang"]
)

# ============================================================ #
# TAB: INTERACTIVE TESTER
# ============================================================ #
with tab_tester:
    section_title("🧪", "Uji Coba Pipeline pada Satu Kalimat")
    sample_text = st.text_area(
        "Masukkan komentar", height=100,
        value="Gw msh dukung chatGPT krn dia dh bntuin gw ngerjain tugas coding, walaupun kadang ngelag bgt 😭",
    )
    if st.button("▶️ Jalankan Pipeline", type="primary"):
        result = pp.full_pipeline(sample_text)

        steps = [
            ("1️⃣ Teks Asli", result["raw"]),
            ("2️⃣ Cleaning & Case Folding", result["clean_text"]),
            ("3️⃣ Normalisasi Slang", result["normalized_text"]),
            ("4️⃣ Tokenisasi", " | ".join(result["tokens"])),
            ("5️⃣ Stopword Removal", " | ".join(result["filtered_tokens"])),
            ("6️⃣ Stemming (Sastrawi)", " | ".join(result["stemmed_tokens"])),
            ("✅ Teks Akhir (untuk TF-IDF)", result["final_text"]),
        ]
        for title, content in steps:
            st.markdown(f"**{title}**")
            st.code(content if content else "(kosong)", language=None)

# ============================================================ #
# TAB: BULK PROCESSING
# ============================================================ #
with tab_bulk:
    section_title("⚙️", "Jalankan Preprocessing untuk Seluruh Dataset")

    stats = db.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("Total Komentar", stats["total"])
    c2.metric("Sudah Diproses", stats["preprocessed"])

    only_new = st.checkbox("Hanya proses komentar yang belum diproses (disarankan)", value=True)
    batch_df = db.fetch_all(preprocessed_only=False) if not only_new else db.fetch_all()
    if only_new:
        batch_df = batch_df[batch_df["is_preprocessed"] == 0]

    st.write(f"Akan memproses **{len(batch_df)}** komentar.")

    if st.button("🚀 Jalankan Preprocessing Sekarang", type="primary", disabled=(len(batch_df) == 0)):

        # === TAMBAHAN MODIFIKASI: Hapus data duplikat & kosong di sini ===
        with st.spinner("Membersihkan data duplikat dan kosong terlebih dahulu..."):
            all_data = db.fetch_all().sort_values("id") # Urutkan dari id terkecil (terlama)
            
            # Deteksi baris yang duplikat atau terlalu pendek
            dup_cid = all_data[all_data.duplicated(subset=["comment_id"], keep="first")]["id"].tolist()
            dup_txt = all_data[all_data.duplicated(subset=["komentar_asli"], keep="first")]["id"].tolist()
            short_txt = all_data[all_data["komentar_asli"].astype(str).str.len() <= 3]["id"].tolist()
            
            # Gabungkan semua id kotor menggunakan Set (agar id tidak double)
            to_delete = list(set(dup_cid + dup_txt + short_txt))
            
            # Hapus dari database
            if len(to_delete) > 0:
                db.delete_many(to_delete)
                st.info(f"Berhasil membuang {len(to_delete)} komentar duplikat/kotor. Data tersisa: {len(all_data) - len(to_delete)} baris.")
            
            # Ambil ulang data (batch_df) yang sudah bersih untuk di-preprocess
            batch_df = db.fetch_all(preprocessed_only=False) if not only_new else db.fetch_all()
            if only_new:
                batch_df = batch_df[batch_df["is_preprocessed"] == 0]
        # ==================================================================

        progress = st.progress(0, text="Memulai...")
        rows_to_update = []
        n = len(batch_df)
        for i, r in enumerate(batch_df.itertuples()):
            result = pp.full_pipeline(r.komentar_asli)
            rows_to_update.append((result["clean_text"], result["normalized_text"], result["final_text"], r.id))
            if (i + 1) % 25 == 0 or (i + 1) == n:
                progress.progress((i + 1) / n, text=f"Memproses {i+1}/{n} komentar...")

        db.bulk_update_preprocessing(rows_to_update)
        st.success(f"✅ Preprocessing selesai untuk {n} komentar.")
        st.rerun()

    st.divider()
    section_title("📋", "Contoh Hasil Preprocessing")
    sample = db.fetch_all(preprocessed_only=True, limit=10)
    if len(sample):
        st.dataframe(
            sample[["komentar_asli", "clean_text", "normalized_text", "final_text"]]
            .rename(columns={
                "komentar_asli": "Asli", "clean_text": "Clean",
                "normalized_text": "Normalized", "final_text": "Final (stemmed)",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Belum ada data yang diproses.")

# ============================================================ #
# TAB: SLANG DICTIONARY (read-only viewer, sesuai file slang_dict.py)
# ============================================================ #
with tab_dict:
    section_title("📖", "Kamus Normalisasi Slang (slang_dict.py)")
    st.caption(f"Total {len(pp.slang_dict)} entri kata tidak baku → kata baku.")

    search_slang = st.text_input("Cari kata slang")
    dict_df = pd.DataFrame(sorted(pp.slang_dict.items()), columns=["Kata Slang", "Kata Baku"])
    if search_slang:
        dict_df = dict_df[dict_df["Kata Slang"].str.contains(search_slang, case=False) |
                           dict_df["Kata Baku"].str.contains(search_slang, case=False)]
    st.dataframe(dict_df, use_container_width=True, hide_index=True, height=420)
