import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from modules import database as db
from modules.auth import require_login, auth_sidebar_footer
from modules.utils import inject_base_style, page_header, section_title

st.set_page_config(page_title="Topic Modeling (LDA)", page_icon="🧩", layout="wide")
inject_base_style()
require_login()
db.init_db()
auth_sidebar_footer()

page_header(
    icon="🧩",
    title="Topic Modeling — Latent Dirichlet Allocation (LDA)",
    subtitle="Mengidentifikasi topik-topik utama yang dibahas dalam komentar berdasarkan hasil preprocessing.",
    tags=["🔑 Kata Dominan", "📊 Distribusi Topik", "💬 Contoh Komentar"],
    gradient="pink",
)

if not db.is_seeded():
    st.warning("Database masih kosong. Muat dataset awal dari halaman Beranda terlebih dahulu.")
    st.stop()

df = db.fetch_all()
corpus_df = df[df["final_text"].notna() & (df["final_text"] != "")]

if len(corpus_df) < 30:
    st.warning("Data hasil preprocessing masih terlalu sedikit. Jalankan halaman **🧹 Preprocessing** terlebih dahulu.")
    st.stop()

st.sidebar.header("⚙️ Konfigurasi LDA")
n_topics = st.sidebar.slider("Jumlah Topik", 2, 10, 3)
n_top_words = st.sidebar.slider("Kata teratas per topik", 5, 20, 10)
max_features = st.sidebar.slider("TF-IDF max_features", 500, 10000, 5000, 500)

run = st.button("🚀 Jalankan LDA", type="primary")

if run:
    with st.spinner("Membangun representasi TF-IDF..."):
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
        X_tfidf = tfidf.fit_transform(corpus_df["final_text"])
        feature_names = tfidf.get_feature_names_out()

    with st.spinner("Melatih model LDA..."):
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_model.fit(X_tfidf)

    st.success("✅ LDA selesai dijalankan.")

    section_title("🔑", "Kata-Kata Dominan per Topik")
    st.caption("📖 Setiap topik direpresentasikan oleh kata-kata yang paling sering muncul bersamaan. "
               "Perhatikan pola kata untuk memberi 'nama' pada tiap topik (mis. topik seputar performa, "
               "harga, atau kemudahan penggunaan).")
    topic_words = {}
    cols = st.columns(min(3, n_topics))
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_words[topic_idx] = top_words
        with cols[topic_idx % len(cols)]:
            st.markdown(f"**Topik {topic_idx + 1}**")
            st.write(", ".join(top_words))

    topic_result = lda_model.transform(X_tfidf)
    dominant_topic = topic_result.argmax(axis=1) + 1

    result_df = corpus_df.copy()
    result_df["Topik"] = dominant_topic

    section_title("📊", "Distribusi Topik")
    st.caption("📖 Menunjukkan berapa banyak komentar yang paling condong (dominan) ke masing-masing topik, "
               "membantu melihat topik mana yang paling banyak dibicarakan.")
    dist = result_df["Topik"].value_counts().sort_index().reset_index()
    dist.columns = ["Topik", "Jumlah Komentar"]
    dist["Topik"] = dist["Topik"].apply(lambda x: f"Topik {x}")
    fig = px.bar(dist, x="Topik", y="Jumlah Komentar", color="Topik", text="Jumlah Komentar")
    fig.update_layout(showlegend=False, height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    section_title("💬", "Contoh Komentar per Topik")
    for i in range(1, n_topics + 1):
        with st.expander(f"Topik {i} — kata kunci: {', '.join(topic_words[i-1][:6])}"):
            examples = result_df[result_df["Topik"] == i]["komentar_asli"].head(5)
            for j, komentar in enumerate(examples, start=1):
                st.write(f"{j}. {komentar}")
else:
    st.info("Atur parameter di sidebar lalu klik **Jalankan LDA** untuk melihat topik-topik yang ditemukan.")
