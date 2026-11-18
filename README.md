# Analisis sentimen dan topik komentar YouTube

Proyek kuliah yang mengolah komentar tentang aplikasi AI menjadi analisis sentimen dan topik. Aplikasi Streamlit meliputi pengelolaan data SQLite, pembersihan dan normalisasi teks, pelabelan berbasis leksikon, serta evaluasi Naive Bayes dan SVM dengan TF-IDF. LDA digunakan untuk eksplorasi topik.

## Menjalankan demo

Gunakan Python 3.11. Dari PowerShell di direktori repositori:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:YOUTUBE_DEMO_PASSWORD = Read-Host 'Tentukan password demo lokal'
.\.venv\Scripts\python -m streamlit run app.py --server.address 127.0.0.1
```

Masuk sebagai `demo` dengan password yang baru Anda tentukan. Muat dataset awal dari sidebar, lalu ikuti halaman Preprocessing, Labeling Sentimen, Pemodelan dan Topic Modeling.

## Data dan evaluasi

`data/demo-comments.xlsx` berisi 48 komentar sintetis tanpa nama pengguna, kanal atau ID video asli. Dataset penelitian menyertakan 6.000 baris dan disimpan terpisah. Hasil pada demo hanya menggambarkan alur aplikasi. Metrik klasifikasi mengukur kesesuaian dengan label leksikon yang dipakai sebagai target.
