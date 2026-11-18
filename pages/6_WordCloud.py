import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from modules import database as db
from modules.auth import require_login, auth_sidebar_footer
from modules.utils import inject_base_style, page_header, section_title

st.set_page_config(page_title="WordCloud & Frekuensi", page_icon="☁️", layout="wide")
inject_base_style()
require_login()
db.init_db()
auth_sidebar_footer()

page_header(
    icon="☁️",
    title="WordCloud & Analisis Frekuensi Kata",
    subtitle="Visualisasi kata yang paling sering muncul setelah preprocessing, secara keseluruhan maupun per sentimen.",
    tags=["☁️ WordCloud", "📊 Frekuensi Kata", "😊😠 Per Sentimen"],
    gradient="blue",
)

if not db.is_seeded():
    st.warning("Database masih kosong. Muat dataset awal dari halaman Beranda terlebih dahulu.")
    st.stop()

df = db.fetch_all()
corpus_df = df[df["final_text"].notna() & (df["final_text"] != "")]

if len(corpus_df) == 0:
    st.warning("Belum ada data hasil preprocessing. Jalankan halaman **🧹 Preprocessing** terlebih dahulu.")
    st.stop()

top_n = st.sidebar.slider("Top-N kata paling sering", 10, 50, 20)
max_words_cloud = st.sidebar.slider("Maksimum kata di WordCloud", 50, 300, 150, 25)


def make_wordcloud(text, colormap):
    if not text.strip():
        return None
    wc = WordCloud(width=1000, height=500, background_color="white",
                    colormap=colormap, max_words=max_words_cloud, random_state=42).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


tab_all, tab_sentiment, tab_freq = st.tabs(["☁️ WordCloud Keseluruhan", "😊😠 WordCloud per Sentimen", "📊 Frekuensi Kata"])

with tab_all:
    st.caption("📖 Semakin besar ukuran kata, semakin sering kata tersebut muncul di seluruh komentar "
               "(setelah dibersihkan & di-stem). Berguna untuk melihat topik dominan secara sekilas.")
    all_text = " ".join(corpus_df["final_text"])
    fig = make_wordcloud(all_text, "viridis")
    if fig:
        st.pyplot(fig)
    else:
        st.info("Tidak cukup teks untuk membuat WordCloud.")

with tab_sentiment:
    st.caption("📖 Bandingkan kata-kata yang paling menonjol pada komentar Positif (hijau) vs. Negatif (merah) "
               "untuk memahami apa yang disukai/dikeluhkan pengguna.")
    labeled_corpus = corpus_df[corpus_df["label"].notna()]
    if len(labeled_corpus) == 0:
        st.info("Belum ada data berlabel. Jalankan halaman **🏷️ Labeling Sentimen** terlebih dahulu.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 😊 Sentimen Positif")
            pos_text = " ".join(labeled_corpus[labeled_corpus["label"] == 1]["final_text"])
            fig_pos = make_wordcloud(pos_text, "Greens")
            if fig_pos:
                st.pyplot(fig_pos)
            else:
                st.caption("Tidak ada data positif.")
        with c2:
            st.markdown("#### 😠 Sentimen Negatif")
            neg_text = " ".join(labeled_corpus[labeled_corpus["label"] == 0]["final_text"])
            fig_neg = make_wordcloud(neg_text, "Reds")
            if fig_neg:
                st.pyplot(fig_neg)
            else:
                st.caption("Tidak ada data negatif.")

with tab_freq:
    st.caption("📖 Diagram batang berikut mengurutkan kata-kata berdasarkan jumlah kemunculan (frekuensi) "
               "tertinggi — pelengkap kuantitatif dari visual WordCloud di atas.")
    all_text = " ".join(corpus_df["final_text"])
    tokens = all_text.split()
    counter = Counter(tokens)
    top_words = counter.most_common(top_n)
    freq_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

    fig = px.bar(freq_df, x="Kata", y="Frekuensi", color="Frekuensi", color_continuous_scale="Blues")
    fig.update_layout(height=450, xaxis_tickangle=-45, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(freq_df, use_container_width=True, hide_index=True)
