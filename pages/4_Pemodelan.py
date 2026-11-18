import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from modules import database as db
from modules.auth import require_login, auth_sidebar_footer
from modules.utils import (
    inject_base_style, save_models, load_models, models_available,
    page_header, section_title,
)

st.set_page_config(page_title="Pemodelan & Evaluasi", page_icon="🤖", layout="wide")
inject_base_style()
require_login()
db.init_db()
auth_sidebar_footer()

page_header(
    icon="🤖",
    title="Pemodelan Klasifikasi Sentimen",
    subtitle="TF-IDF Feature Extraction → Naive Bayes vs. Support Vector Machine (LinearSVC), "
             "lengkap dengan evaluasi dan perbandingan performa model.",
    tags=["🔠 TF-IDF", "🧠 Naive Bayes", "📐 SVM", "📊 Confusion Matrix"],
    gradient="dark",
)

if not db.is_seeded():
    st.warning("Database masih kosong. Muat dataset awal dari halaman Beranda terlebih dahulu.")
    st.stop()

stats = db.get_stats()
if stats["labeled"] < 20:
    st.warning("Data berlabel masih terlalu sedikit (minimal ±20 disarankan). "
               "Jalankan halaman **🏷️ Labeling Sentimen** terlebih dahulu.")
    st.stop()

# ------------------------------------------------------------------ #
# KONFIGURASI
# ------------------------------------------------------------------ #
st.sidebar.header("⚙️ Konfigurasi Pemodelan")
test_size = st.sidebar.slider("Proporsi Data Uji", 0.1, 0.4, 0.2, 0.05)
max_features = st.sidebar.slider("TF-IDF max_features", 500, 10000, 5000, 500)
ngram_max = st.sidebar.selectbox("TF-IDF ngram_range", [(1, 1), (1, 2), (1, 3)], index=1,
                                  format_func=lambda x: f"{x}")
min_df = st.sidebar.slider("TF-IDF min_df", 1, 10, 2)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

df = db.fetch_all()
labeled_df = df[df["label"].notna() & df["final_text"].notna() & (df["final_text"] != "")]

st.write(f"Data siap latih: **{len(labeled_df)}** komentar berlabel dan sudah diproses.")

train_btn = st.button("🚀 Latih Model (TF-IDF + Naive Bayes + SVM)", type="primary")

if train_btn:
    X = labeled_df["final_text"]
    y = labeled_df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    with st.spinner("Membangun fitur TF-IDF..."):
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_max, min_df=min_df)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

    with st.spinner("Melatih Naive Bayes..."):
        nb_model = MultinomialNB()
        nb_model.fit(X_train_tfidf, y_train)
        y_pred_nb = nb_model.predict(X_test_tfidf)

    with st.spinner("Melatih SVM (LinearSVC)..."):
        svm_model = LinearSVC(random_state=int(random_state))
        svm_model.fit(X_train_tfidf, y_train)
        y_pred_svm = svm_model.predict(X_test_tfidf)

    save_models(tfidf, nb_model, svm_model, meta={
        "test_size": test_size, "max_features": max_features,
        "ngram_range": ngram_max, "min_df": min_df, "n_train": len(X_train), "n_test": len(X_test),
    })

    st.session_state["last_eval"] = {
        "y_test": y_test, "y_pred_nb": y_pred_nb, "y_pred_svm": y_pred_svm,
        "X_test": X_test, "X_train_tfidf": X_train_tfidf, "feature_names": tfidf.get_feature_names_out(),
    }
    st.success("✅ Model berhasil dilatih dan disimpan.")

# ------------------------------------------------------------------ #
# EVALUASI (memakai hasil training terbaru dalam sesi ini)
# ------------------------------------------------------------------ #
if "last_eval" in st.session_state:
    ev = st.session_state["last_eval"]
    y_test, y_pred_nb, y_pred_svm = ev["y_test"], ev["y_pred_nb"], ev["y_pred_svm"]

    def metrics_of(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
        }

    m_nb = metrics_of(y_test, y_pred_nb)
    m_svm = metrics_of(y_test, y_pred_svm)

    section_title("📊", "Perbandingan Performa Model")
    comp_df = pd.DataFrame({
        "Model": ["Naive Bayes", "Support Vector Machine"],
        "Accuracy (%)": [round(m_nb["Accuracy"] * 100, 2), round(m_svm["Accuracy"] * 100, 2)],
        "Precision (%)": [round(m_nb["Precision"] * 100, 2), round(m_svm["Precision"] * 100, 2)],
        "Recall (%)": [round(m_nb["Recall"] * 100, 2), round(m_svm["Recall"] * 100, 2)],
        "F1-Score (%)": [round(m_nb["F1-Score"] * 100, 2), round(m_svm["F1-Score"] * 100, 2)],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption("📖 **Accuracy**: persentase prediksi yang benar dari total data uji. **Precision**: dari semua yang "
               "diprediksi Positif, berapa persen yang benar-benar positif. **Recall**: dari semua data yang "
               "sebenarnya Positif, berapa persen yang berhasil terdeteksi. **F1-Score**: rata-rata harmonis "
               "Precision & Recall — metrik utama untuk membandingkan model secara adil.")

    fig = px.bar(
        comp_df.melt(id_vars="Model", var_name="Metrik", value_name="Nilai (%)"),
        x="Metrik", y="Nilai (%)", color="Model", barmode="group", text="Nilai (%)",
    )
    fig.update_layout(height=420, yaxis_range=[0, 100], plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    best = "Naive Bayes" if m_nb["F1-Score"] >= m_svm["F1-Score"] else "Support Vector Machine"
    st.success(f"🏆 Model dengan F1-Score terbaik pada percobaan ini: **{best}**")

    tab_nb, tab_svm = st.tabs(["📗 Detail Naive Bayes", "📘 Detail SVM"])

    with tab_nb:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m_nb['Accuracy']*100:.2f}%")
        c2.metric("Precision", f"{m_nb['Precision']*100:.2f}%")
        c3.metric("Recall", f"{m_nb['Recall']*100:.2f}%")
        c4.metric("F1-Score", f"{m_nb['F1-Score']*100:.2f}%")

        st.text("Classification Report")
        st.code(classification_report(y_test, y_pred_nb, target_names=["Negatif", "Positif"]))

        st.caption("📖 **Confusion Matrix**: baris = label sebenarnya (Actual), kolom = label hasil prediksi "
                   "(Predicted). Diagonal (kiri-atas ke kanan-bawah) menunjukkan prediksi yang benar — semakin "
                   "besar angka di diagonal, semakin baik model.")
        cm = confusion_matrix(y_test, y_pred_nb)
        fig_cm, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Naive Bayes")
        st.pyplot(fig_cm)

    with tab_svm:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m_svm['Accuracy']*100:.2f}%")
        c2.metric("Precision", f"{m_svm['Precision']*100:.2f}%")
        c3.metric("Recall", f"{m_svm['Recall']*100:.2f}%")
        c4.metric("F1-Score", f"{m_svm['F1-Score']*100:.2f}%")

        st.text("Classification Report")
        st.code(classification_report(y_test, y_pred_svm, target_names=["Negatif", "Positif"]))

        st.caption("📖 **Confusion Matrix**: baris = label sebenarnya (Actual), kolom = label hasil prediksi "
                   "(Predicted). Diagonal menunjukkan prediksi yang benar.")
        cm2 = confusion_matrix(y_test, y_pred_svm)
        fig_cm2, ax2 = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Purples",
                    xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"], ax=ax2)
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
        ax2.set_title("Confusion Matrix - SVM")
        st.pyplot(fig_cm2)

elif models_available():
    st.info("Menampilkan metadata model tersimpan dari pelatihan sebelumnya. Latih ulang untuk melihat evaluasi lengkap sesi ini.")
    saved = load_models()
    st.json(saved["meta"])
else:
    st.info("Klik tombol **Latih Model** di atas untuk memulai pemodelan.")
