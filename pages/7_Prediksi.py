import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt

import streamlit as st

from modules import database as db
from modules.auth import require_login, auth_sidebar_footer
from modules import preprocessing as pp
from modules.utils import inject_base_style, load_models, models_available, page_header, section_title

st.set_page_config(page_title="Prediksi Komentar Baru", page_icon="🔮", layout="wide")
inject_base_style()
require_login()
db.init_db()
auth_sidebar_footer()

page_header(
    icon="🔮",
    title="Prediksi Sentimen Komentar Baru",
    subtitle="Masukkan komentar baru untuk diproses lewat pipeline yang sama, lalu diklasifikasikan oleh model terlatih.",
    tags=["🧹 Auto-Preprocess", "🧠 Naive Bayes", "📐 SVM", "📥 Prediksi Massal"],
    gradient="pink",
)

if not models_available():
    st.warning("Model belum dilatih. Kunjungi halaman **🤖 Pemodelan & Evaluasi** terlebih dahulu.")
    st.stop()

saved = load_models()
vectorizer = saved["vectorizer"]
nb_model = saved["nb_model"]
svm_model = saved["svm_model"]
meta = saved["meta"]

with st.expander("ℹ️ Informasi model yang sedang digunakan"):
    st.json(meta)

model_choice = st.radio("Pilih model untuk prediksi", ["Naive Bayes", "Support Vector Machine", "Keduanya (bandingkan)"],
                         horizontal=True)

text_input = st.text_area(
    "Tulis komentar tentang aplikasi AI (ChatGPT, Gemini, dll.)", height=120,
    placeholder="Contoh: Aplikasi ini keren banget, ngebantu bgt buat ngerjain tugas coding!",
)

if st.button("🔍 Prediksi Sentimen", type="primary", disabled=not text_input.strip()):
    result = pp.full_pipeline(text_input)
    final_text = result["final_text"]

    if not final_text.strip():
        st.error("Setelah preprocessing, teks menjadi kosong (mungkin hanya berisi stopword/simbol). Coba komentar lain.")
    else:
        X_new = vectorizer.transform([final_text])

        def show_result(name, model, color):
            pred = model.predict(X_new)[0]
            label_text = "😊 POSITIF" if pred == 1 else "😠 NEGATIF"
            st.markdown(f"### {name}: {label_text}")

        section_title("🎯", "Hasil Prediksi")
        if model_choice in ("Naive Bayes", "Keduanya (bandingkan)"):
            pred_nb = nb_model.predict(X_new)[0]
            st.metric("Naive Bayes", "😊 Positif" if pred_nb == 1 else "😠 Negatif")

        if model_choice in ("Support Vector Machine", "Keduanya (bandingkan)"):
            pred_svm = svm_model.predict(X_new)[0]
            st.metric("Support Vector Machine", "😊 Positif" if pred_svm == 1 else "😠 Negatif")

        with st.expander("🔎 Lihat tahapan preprocessing"):
            st.write("**Teks Asli:**", result["raw"])
            st.write("**Setelah Cleaning:**", result["clean_text"])
            st.write("**Setelah Normalisasi Slang:**", result["normalized_text"])
            st.write("**Token (setelah stopword removal):**", result["filtered_tokens"])
            st.write("**Setelah Stemming:**", result["stemmed_tokens"])
            st.write("**Teks Final (input ke TF-IDF):**", final_text)

        st.divider()
        section_title("💾", "Simpan Komentar Ini ke Database?")
        st.caption("Komentar beserta hasil preprocessing & prediksi (model utama: yang dipilih di atas) akan ditambahkan ke dataset melalui CRUD.")

        final_label = None
        if model_choice == "Naive Bayes":
            final_label = int(nb_model.predict(X_new)[0])
        elif model_choice == "Support Vector Machine":
            final_label = int(svm_model.predict(X_new)[0])
        else:
            final_label = int(svm_model.predict(X_new)[0])

        if st.button("💾 Simpan ke Database"):
            new_id = db.insert_comment({
                "aplikasi": "Prediksi Manual",
                "video_id": "prediction",
                "judul_video": "",
                "channel": "",
                "tanggal_video": str(dt.date.today()),
                "keyword_sumber": "manual-prediction",
                "komentar_asli": text_input.strip(),
                "author": "@prediksi_manual",
                "tanggal_komentar": str(dt.date.today()),
                "like_count": 0,
                "total_reply_count": 0,
            })
            db.update_preprocessing_row(new_id, result["clean_text"], result["normalized_text"], final_text)
            db.bulk_update_labels([(final_label, new_id)])
            st.success(f"✅ Komentar disimpan dengan ID #{new_id} dan sudah berlabel otomatis.")

st.divider()
section_title("📥", "Prediksi Massal (Bulk)")
st.caption("Unggah file CSV/Excel berkolom 'komentar_asli' untuk diprediksi sekaligus.")
bulk_file = st.file_uploader("Unggah file", type=["csv", "xlsx"], key="bulk_predict")

if bulk_file is not None:
    import pandas as pd
    bulk_df = pd.read_csv(bulk_file) if bulk_file.name.endswith(".csv") else pd.read_excel(bulk_file)

    if "komentar_asli" not in bulk_df.columns:
        st.error("File harus memiliki kolom 'komentar_asli'.")
    else:
        if st.button("🚀 Jalankan Prediksi Massal"):
            progress = st.progress(0, text="Memproses...")
            preds = []
            n = len(bulk_df)
            for i, text in enumerate(bulk_df["komentar_asli"].astype(str)):
                ft = pp.full_pipeline(text)["final_text"]
                if ft.strip():
                    Xn = vectorizer.transform([ft])
                    p = int(svm_model.predict(Xn)[0])
                else:
                    p = None
                preds.append("Positif" if p == 1 else ("Negatif" if p == 0 else "Tidak dapat diprediksi"))
                if (i + 1) % 10 == 0 or (i + 1) == n:
                    progress.progress((i + 1) / n, text=f"{i+1}/{n}")
            bulk_df["Prediksi Sentimen (SVM)"] = preds
            st.dataframe(bulk_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Unduh hasil prediksi (CSV)",
                data=bulk_df.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_massal.csv",
                mime="text/csv",
            )
