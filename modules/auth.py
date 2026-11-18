"""
================================================================================
MODULE: auth.py
Sistem autentikasi sederhana berbasis session_state untuk melindungi seluruh
halaman aplikasi. Cocok untuk kebutuhan demo/skripsi (bukan autentikasi
tingkat production) — kredensial diatur lewat variabel lingkungan.
================================================================================
"""

import os
from hmac import compare_digest

import streamlit as st
from modules import database as db

def local_users():
    user = os.environ.get("YOUTUBE_DEMO_USER", "demo").strip().lower()
    password = os.environ.get("YOUTUBE_DEMO_PASSWORD")
    return {user: {"password": password, "name": "Pengguna demo", "role": "demo"}} if password else {}


def _login_screen():
    left, mid, right = st.columns([1, 1.3, 1])
    with mid:
        st.markdown("""
            <div class="hero-banner" style="--hero-gradient: linear-gradient(135deg,#0f172a,#1e293b,#334155); text-align:center; margin-top:40px;">
                <div class="hero-icon" style="margin:0 auto 10px auto;">🔐</div>
                <p class="hero-title" style="text-align:center;">Masuk ke Sistem</p>
                <p class="hero-subtitle" style="text-align:center; margin:6px auto 0 auto;">
                    Dashboard Analisis Sentimen Komentar YouTube tentang AI
                </p>
            </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Username", placeholder="admin")
            password = st.text_input("🔑 Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("🚀 Masuk", type="primary", use_container_width=True)

            if submitted:
                user = local_users().get(username.strip().lower())
                if user and compare_digest(user["password"].encode(), password.encode()):
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip().lower()
                    st.session_state.user_name = user["name"]
                    st.session_state.user_role = user["role"]
                    st.success("Login berhasil! Memuat sistem...")
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah. Silakan coba lagi.")

        st.caption("Gunakan akun demo yang Anda atur di komputer sendiri.")


def require_login():
    """Dipanggil di baris paling atas setiap halaman. Menghentikan render
    (st.stop) apabila pengguna belum login."""
    if not st.session_state.get("authenticated"):
        _login_screen()
        st.stop()


def auth_sidebar_footer():
    """Menampilkan Header, Menu Navigasi, dan Footer Logout secara global."""
    with st.sidebar:
        
        # === 1. BAGIAN LOGO & JUDUL ===
        st.markdown("""
            <div style="text-align:center; padding: 6px 0 14px 0;">
                <div style="font-size:2.6rem; line-height:1;">🤖</div>
                <!-- Gunakan var(--text-color) agar otomatis mengikuti tema -->
                <div style="font-family:'Poppins',sans-serif; font-weight:700; font-size:1.15rem; color:var(--text-color);">
                    Sentimen AI YouTube
                </div>
                <div style="font-size:0.78rem; color:var(--text-color); opacity: 0.7;">
                    Lexicon · TF-IDF · Naive Bayes · SVM · LDA
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.divider()

        # === 2. BAGIAN STATUS DATABASE ===
        if not db.is_seeded():
            st.warning("Database masih kosong.")
            if st.button("📥 Muat dataset awal (Excel)", use_container_width=True, type="primary"):
                with st.spinner("Memuat dataset dari file Excel (data mentah, tanpa preprocessing)..."):
                    n = db.seed_from_excel()
                st.success(f"{n} komentar berhasil dimuat apa adanya (raw), sama persis seperti file Excel sumber.")
                st.info("💡 Buka halaman **Preprocessing** untuk mulai membersihkan & menormalisasi teks.")
                st.rerun()
        else:
            st.success("Database siap digunakan ✅")
            with st.expander("⚠️ Reset Database"):
                st.caption("Menghapus seluruh perubahan CRUD dan memuat ulang dari file Excel sumber (data mentah, belum diproses).")
                confirm = st.checkbox("Saya yakin ingin reset total")
                if st.button("🔄 Reset ke data awal", use_container_width=True, disabled=not confirm):
                    with st.spinner("Mereset database..."):
                        n = db.reset_database()
                    st.success(f"Database direset ({n} baris, data mentah).")
                    st.rerun()
        st.divider()

        # === 3. BAGIAN MENU NAVIGASI KUSTOM ===
        st.markdown('<p style="font-size:0.72rem; font-weight:700; color:var(--text-color); opacity: 0.6; letter-spacing:1px; margin-bottom:2px;">MENU NAVIGASI</p>', unsafe_allow_html=True)
        st.page_link("app.py", label="🏠 Beranda / Dashboard")
        st.page_link("pages/1_Kelola_Data.py", label="🗂️ Kelola Data (CRUD)")
        st.page_link("pages/2_Preprocessing.py", label="🧹 Preprocessing Teks")
        st.page_link("pages/3_Labeling_Sentimen.py", label="🏷️ Labeling Sentimen")
        st.page_link("pages/4_Pemodelan.py", label="🤖 Pemodelan & Evaluasi")
        st.page_link("pages/5_Topic_Modeling.py", label="🧩 Topic Modeling (LDA)")
        st.page_link("pages/6_WordCloud.py", label="☁️ WordCloud & Frekuensi")
        st.page_link("pages/7_Prediksi.py", label="🔮 Prediksi Komentar Baru")
        st.divider()

        # === 4. BAGIAN PROFIL & LOGOUT ===
        st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; padding:8px 4px;">
                <div style="width:38px; height:38px; border-radius:50%;
                            background:linear-gradient(135deg,#6a5cff,#8e7bff);
                            display:flex; align-items:center; justify-content:center;
                            color:white; font-weight:700; font-size:1rem;">
                    {st.session_state.get('user_name', 'U')[0].upper()}
                </div>
                <div>
                    <div style="font-weight:600; font-size:0.88rem;">{st.session_state.get('user_name', 'Pengguna')}</div>
                    <div style="font-size:0.72rem; color:#888;">{st.session_state.get('user_role', '')}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            for k in ["authenticated", "username", "user_name", "user_role"]:
                st.session_state.pop(k, None)
            st.rerun()
