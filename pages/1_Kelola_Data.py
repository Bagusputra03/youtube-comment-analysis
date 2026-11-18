import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import datetime as dt

import pandas as pd
import streamlit as st

from modules import database as db
from modules.auth import require_login, auth_sidebar_footer
from modules.utils import inject_base_style, page_header, section_title, metric_card, info_box

st.set_page_config(page_title="Kelola Data (CRUD)", page_icon="🗂️", layout="wide")
inject_base_style()
require_login()
db.init_db()
auth_sidebar_footer()

page_header(
    icon="🗂️",
    title="Kelola Data Komentar (CRUD)",
    subtitle="Create · Read · Update · Delete langsung terhadap dataset komentar YouTube — "
             "semua perubahan tersimpan permanen di database SQLite.",
    tags=["🔍 Cari & Filter", "➕ Tambah", "✏️ Edit", "🗑️ Hapus", "📤 Import/Export"],
    gradient="blue",
)

if not db.is_seeded():
    st.warning("Database masih kosong. Muat dataset awal dari halaman Beranda terlebih dahulu.")
    st.stop()

stats = db.get_stats()
c1, c2, c3, c4 = st.columns(4)
metric_card(c1, "🗂️", "Total Data", f"{stats['total']:,}", gradient="blue")
metric_card(c2, "🧹", "Sudah Diproses", f"{stats['preprocessed']:,}", gradient="purple")
metric_card(c3, "🏷️", "Sudah Dilabeli", f"{stats['labeled']:,}", gradient="orange")
metric_card(c4, "📱", "Jumlah Aplikasi", f"{stats['n_apps']:,}", gradient="green")
st.write("")

tab_read, tab_create, tab_update, tab_delete, tab_io = st.tabs(
    ["🔍 Lihat & Cari", "➕ Tambah Data", "✏️ Edit Data", "🗑️ Hapus Data", "📤 Import / Export"]
)

# ============================================================ #
# TAB: READ (lihat, cari, filter, pagination)
# ============================================================ #
with tab_read:
    section_title("🔍", "Pencarian & Filter Data")

    apps = ["Semua"] + db.distinct_values("aplikasi")
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        search = st.text_input("Cari komentar / penulis / judul video", key="search_read")
    with f2:
        aplikasi_f = st.selectbox("Aplikasi", apps, key="aplikasi_read")
    with f3:
        label_f = st.selectbox("Status Label", ["Semua", "Positif (1)", "Negatif (0)", "Belum Dilabeli"], key="label_read")
    with f4:
        page_size = st.selectbox("Baris / halaman", [10, 25, 50, 100], index=1, key="pagesize_read")

    label_param = "Semua"
    if label_f == "Positif (1)":
        label_param = 1
    elif label_f == "Negatif (0)":
        label_param = 0
    elif label_f == "Belum Dilabeli":
        label_param = None

    if label_f == "Belum Dilabeli":
        full_df = db.fetch_all(search=search or None, aplikasi=aplikasi_f)
        full_df = full_df[full_df["label"].isna()]
    else:
        full_df = db.fetch_all(search=search or None, aplikasi=aplikasi_f, label=label_param)

    total_rows = len(full_df)
    total_pages = max(1, -(-total_rows // page_size))

    st.write(f"Ditemukan **{total_rows}** komentar.")
    page_num = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1, key="page_read")

    start = (page_num - 1) * page_size
    end = start + page_size
    view_df = full_df.iloc[start:end].copy()

    if len(view_df) == 0:
        st.info("Tidak ada data yang cocok dengan filter.")
    else:
        display_cols = ["id", "aplikasi", "author", "komentar_asli", "like_count",
                         "total_reply_count", "label", "is_preprocessed", "tanggal_komentar"]
        show = view_df[display_cols].rename(columns={
            "id": "ID", "aplikasi": "Aplikasi", "author": "Penulis",
            "komentar_asli": "Komentar", "like_count": "Suka",
            "total_reply_count": "Balasan", "label": "Label",
            "is_preprocessed": "Preprocessed", "tanggal_komentar": "Tanggal",
        })
        show["Label"] = show["Label"].map({1: "😊 Positif", 0: "😠 Negatif"}).fillna("— Belum dilabeli")
        show["Preprocessed"] = show["Preprocessed"].map({1: "✅", 0: "❌"})
        st.dataframe(show, use_container_width=True, hide_index=True, height=420)

        st.download_button(
            "⬇️ Unduh hasil filter ini (CSV)",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="komentar_filtered.csv",
            mime="text/csv",
        )

# ============================================================ #
# TAB: CREATE
# ============================================================ #
with tab_create:
    section_title("➕", "Tambah Komentar Baru")
    st.caption("Menambahkan baris data baru secara manual ke dalam dataset (misalnya komentar yang ditemukan di lapangan).")

    with st.form("form_create", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            aplikasi = st.selectbox("Aplikasi AI", db.distinct_values("aplikasi") or ["ChatGPT", "Gemini", "Perplexity"])
            author = st.text_input("Nama Penulis Komentar", value="@pengguna_baru")
            like_count = st.number_input("Jumlah Suka (Like)", min_value=0, value=0)
            total_reply_count = st.number_input("Jumlah Balasan", min_value=0, value=0)
        with c2:
            judul_video = st.text_input("Judul Video", value="")
            channel = st.text_input("Channel", value="")
            keyword_sumber = st.text_input("Keyword Sumber", value="")
            tanggal_komentar = st.date_input("Tanggal Komentar", value=dt.date.today())

        komentar_asli = st.text_area("Isi Komentar *", height=120, placeholder="Tulis komentar di sini...")

        submitted = st.form_submit_button("💾 Simpan Komentar", type="primary", use_container_width=True)
        if submitted:
            if not komentar_asli.strip():
                st.error("Isi komentar tidak boleh kosong.")
            else:
                new_id = db.insert_comment({
                    "aplikasi": aplikasi,
                    "video_id": "manual",
                    "judul_video": judul_video,
                    "channel": channel,
                    "tanggal_video": str(dt.date.today()),
                    "keyword_sumber": keyword_sumber,
                    "komentar_asli": komentar_asli.strip(),
                    "author": author,
                    "tanggal_komentar": str(tanggal_komentar),
                    "like_count": int(like_count),
                    "total_reply_count": int(total_reply_count),
                })
                st.success(f"✅ Komentar baru berhasil disimpan dengan ID **{new_id}**.")

# ============================================================ #
# TAB: UPDATE
# ============================================================ #
with tab_update:
    section_title("✏️", "Edit Komentar")

    search_edit = st.text_input("Cari komentar yang ingin diedit (ID / isi komentar / penulis)", key="search_edit")
    edit_df = db.fetch_all(search=search_edit or None, limit=25)

    if len(edit_df) == 0:
        st.info("Tidak ditemukan data. Coba kata kunci lain.")
    else:
        options = {f'#{r.id} — {r.author} — "{r.komentar_asli[:60]}..."': r.id for r in edit_df.itertuples()}
        selected_label = st.selectbox("Pilih komentar", list(options.keys()))
        selected_id = options[selected_label]
        row = db.fetch_by_id(selected_id)

        with st.form("form_update"):
            c1, c2 = st.columns(2)
            with c1:
                aplikasi = st.text_input("Aplikasi AI", value=row["aplikasi"] or "")
                author = st.text_input("Penulis", value=row["author"] or "")
                like_count = st.number_input("Suka", min_value=0, value=int(row["like_count"] or 0))
                total_reply_count = st.number_input("Balasan", min_value=0, value=int(row["total_reply_count"] or 0))
            with c2:
                judul_video = st.text_input("Judul Video", value=row["judul_video"] or "")
                channel = st.text_input("Channel", value=row["channel"] or "")
                keyword_sumber = st.text_input("Keyword Sumber", value=row["keyword_sumber"] or "")

            komentar_asli = st.text_area("Isi Komentar *", value=row["komentar_asli"] or "", height=120)

            st.caption("⚠️ Mengubah isi komentar akan menghapus hasil preprocessing & label sebelumnya "
                       "(harus dijalankan ulang).")

            submitted = st.form_submit_button("💾 Simpan Perubahan", type="primary", use_container_width=True)
            if submitted:
                if not komentar_asli.strip():
                    st.error("Isi komentar tidak boleh kosong.")
                else:
                    db.update_comment(selected_id, {
                        "aplikasi": aplikasi, "author": author,
                        "like_count": int(like_count), "total_reply_count": int(total_reply_count),
                        "judul_video": judul_video, "channel": channel,
                        "keyword_sumber": keyword_sumber, "komentar_asli": komentar_asli.strip(),
                    })
                    st.success(f"✅ Komentar #{selected_id} berhasil diperbarui.")
                    st.rerun()

        with st.expander("🔎 Lihat detail hasil preprocessing / label saat ini"):
            st.json({
                "clean_text": row["clean_text"],
                "normalized_text": row["normalized_text"],
                "final_text": row["final_text"],
                "label": row["label"],
            })

# ============================================================ #
# TAB: DELETE
# ============================================================ #
with tab_delete:
    section_title("🗑️", "Hapus Komentar Terpilih")

    search_del = st.text_input("Cari komentar yang ingin dihapus", key="search_del")
    del_df = db.fetch_all(search=search_del or None, limit=100)

    if len(del_df) == 0:
        st.info("Tidak ditemukan data.")
    else:
        show_del = del_df[["id", "aplikasi", "author", "komentar_asli", "label"]].copy()
        show_del.insert(0, "Hapus?", False)
        edited = st.data_editor(
            show_del, use_container_width=True, hide_index=True, height=380,
            column_config={"Hapus?": st.column_config.CheckboxColumn(required=True)},
            disabled=["id", "aplikasi", "author", "komentar_asli", "label"],
            key="delete_editor",
        )

        to_delete = edited[edited["Hapus?"]]["id"].tolist()
        st.write(f"Dipilih untuk dihapus: **{len(to_delete)}** baris")

        confirm_del = st.checkbox("Saya yakin ingin menghapus data terpilih (tidak dapat dibatalkan)")
        if st.button("🗑️ Hapus Data Terpilih", type="primary", disabled=(len(to_delete) == 0 or not confirm_del)):
            db.delete_many(to_delete)
            st.success(f"✅ {len(to_delete)} komentar berhasil dihapus.")
            st.rerun()

    # ======================================================== #
    # DANGER ZONE — HAPUS SEMUA DATA SEKALIGUS
    # ======================================================== #
    st.divider()
    st.markdown("""
        <div class="danger-zone">
            <h4>🚨 Danger Zone — Hapus Semua Data</h4>
            <p style="margin:0; color:#7a2020; font-size:0.88rem;">
                Aksi ini akan <b>menghapus seluruh baris data</b> di database (bukan hanya hasil filter),
                termasuk hasil preprocessing dan label yang sudah dibuat. Tindakan ini
                <b>tidak dapat dibatalkan</b>. Gunakan hanya jika Anda benar-benar ingin mengosongkan dataset.
            </p>
        </div>
    """, unsafe_allow_html=True)

    current_total = db.get_stats()["total"]
    st.write("")
    dc1, dc2 = st.columns([2, 1])
    with dc1:
        confirm_text = st.text_input(
            f"Ketik **HAPUS SEMUA** untuk mengonfirmasi penghapusan seluruh {current_total:,} data",
            key="confirm_delete_all", placeholder="HAPUS SEMUA",
        )
    with dc2:
        st.write("")
        st.write("")
        confirm_checkbox = st.checkbox("Saya paham risikonya", key="confirm_delete_all_checkbox")

    delete_all_ready = (confirm_text.strip().upper() == "HAPUS SEMUA") and confirm_checkbox and current_total > 0

    if st.button("🔥 HAPUS SEMUA DATA SEKARANG", type="primary",
                 disabled=not delete_all_ready, use_container_width=True):
        n_deleted = db.delete_all_comments()
        st.success(f"✅ Seluruh data ({n_deleted:,} baris) berhasil dihapus. Database sekarang kosong.")
        st.info("Gunakan tombol **📥 Muat dataset awal (Excel)** di sidebar Beranda, atau tab "
                "**➕ Tambah Data** / **📤 Import / Export** di halaman ini, untuk mengisi kembali data.")
        st.rerun()

    if current_total == 0:
        st.caption("Database sudah kosong — tidak ada yang perlu dihapus.")

# ============================================================ #
# TAB: IMPORT / EXPORT
# ============================================================ #
with tab_io:
    section_title("📥", "Import Data Tambahan")
    uploaded = st.file_uploader("Unggah file Excel/CSV tambahan (kolom minimal: komentar_asli)", type=["xlsx", "csv"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                new_df = pd.read_csv(uploaded)
            else:
                new_df = pd.read_excel(uploaded)
            st.dataframe(new_df.head(10), use_container_width=True)

            if "komentar_asli" not in new_df.columns:
                st.error("File harus memiliki kolom 'komentar_asli'.")
            else:
                if st.button("📥 Impor Semua Baris ke Database", type="primary"):
                    progress = st.progress(0, text="Mengimpor data...")
                    n = len(new_df)
                    for i, r in enumerate(new_df.to_dict(orient="records")):
                        db.insert_comment({
                            "aplikasi": r.get("aplikasi", "Lainnya"),
                            "video_id": r.get("video_id", "import"),
                            "judul_video": r.get("judul_video", ""),
                            "channel": r.get("channel", ""),
                            "tanggal_video": str(r.get("tanggal_video", "")),
                            "keyword_sumber": r.get("keyword_sumber", ""),
                            "komentar_asli": str(r.get("komentar_asli", "")),
                            "author": r.get("author", "@import"),
                            "tanggal_komentar": str(r.get("tanggal_komentar", dt.date.today())),
                            "like_count": int(r.get("like_count", 0) or 0),
                            "total_reply_count": int(r.get("total_reply_count", 0) or 0),
                        })
                        progress.progress((i + 1) / n, text=f"Mengimpor {i+1}/{n}")
                    st.success(f"✅ {n} baris berhasil diimpor.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    st.divider()
    section_title("📤", "Export Seluruh Database")
    full_export = db.fetch_all()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Unduh sebagai CSV",
            data=full_export.to_csv(index=False).encode("utf-8"),
            file_name="dataset_komentar_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            full_export.to_excel(writer, index=False, sheet_name="komentar")
        st.download_button(
            "⬇️ Unduh sebagai Excel",
            data=buffer.getvalue(),
            file_name="dataset_komentar_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
