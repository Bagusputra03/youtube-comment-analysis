"""
================================================================================
MODULE: preprocessing.py
Implementasi pipeline preprocessing teks Bahasa Indonesia, diambil dan
dirapikan dari notebook SKRIPSI.ipynb:

    1. Cleaning & case folding
    2. Normalisasi slang (slang_dict.py)
    3. Tokenisasi
    4. Stopword removal (custom, mempertahankan kata negasi)
    5. Stemming (Sastrawi, dengan skip-list utk istilah AI/Inggris)

Catatan: NLTK diganti dengan tokenizer regex + Sastrawi StopWordRemover agar
aplikasi tidak bergantung pada unduhan data eksternal (nltk.download) saat
dijalankan, sehingga lebih andal untuk dijalankan di mana saja.
================================================================================
"""

import re
import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from .slang_dict import slang_dict

# ------------------------------------------------------------------ #
# INISIALISASI SASTRAWI (sekali saja, lalu dipakai ulang / cache)
# ------------------------------------------------------------------ #
_stemmer_factory = StemmerFactory()
_stemmer = _stemmer_factory.create_stemmer()

_stopword_factory = StopWordRemoverFactory()
BASE_STOPWORDS = set(_stopword_factory.get_stop_words())

# Kata yang WAJIB dipertahankan (negasi & intensitas) — sesuai notebook
KEEP_WORDS = {"tidak", "tak", "bukan", "belum", "sudah", "sangat"}
STOPWORDS = BASE_STOPWORDS - KEEP_WORDS

# Istilah AI / Inggris yang tidak perlu di-stem
SKIP_STEM_WORDS = {
    "chatgpt", "gemini", "perplexity", "ai", "prompt",
    "api", "youtube", "google", "programmer", "code",
    "coding", "review", "video", "system",
}

_STEM_CACHE = {}


# ------------------------------------------------------------------ #
# STEP 1: CLEANING & CASE FOLDING
# ------------------------------------------------------------------ #
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------------ #
# STEP 2: NORMALISASI SLANG
# ------------------------------------------------------------------ #
def normalize_text(text: str, extra_slang: dict = None) -> str:
    mapping = slang_dict
    if extra_slang:
        mapping = {**slang_dict, **extra_slang}

    words = text.split()
    normalized = []
    for word in words:
        word = word.strip(string.punctuation)
        normalized.append(mapping.get(word, word))
    return " ".join(normalized)


# ------------------------------------------------------------------ #
# STEP 3: TOKENISASI (regex, tidak butuh nltk.download)
# ------------------------------------------------------------------ #
def tokenize_text(text: str) -> list:
    return re.findall(r"[a-zA-Z]+", text)


# ------------------------------------------------------------------ #
# STEP 4: STOPWORD REMOVAL (custom, mempertahankan negasi)
# ------------------------------------------------------------------ #
def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS]


# ------------------------------------------------------------------ #
# STEP 5: STEMMING (Sastrawi + skip-list + cache)
# ------------------------------------------------------------------ #
def smart_stemming(tokens: list) -> list:
    result = []
    for word in tokens:
        if word in SKIP_STEM_WORDS:
            result.append(word)
            continue
        if word in _STEM_CACHE:
            result.append(_STEM_CACHE[word])
        else:
            stemmed = _stemmer.stem(word)
            _STEM_CACHE[word] = stemmed
            result.append(stemmed)
    return result


# ------------------------------------------------------------------ #
# PIPELINE LENGKAP (untuk 1 buah teks / komentar)
# ------------------------------------------------------------------ #
def full_pipeline(raw_text: str, extra_slang: dict = None) -> dict:
    """Menjalankan seluruh pipeline pada satu teks dan mengembalikan setiap
    tahapan agar bisa ditampilkan secara transparan di UI."""
    step1 = clean_text(raw_text)
    step2 = normalize_text(step1, extra_slang=extra_slang)
    step3 = tokenize_text(step2)
    step4 = remove_stopwords(step3)
    step5 = smart_stemming(step4)
    final_text = " ".join(step5)

    return {
        "raw": raw_text,
        "clean_text": step1,
        "normalized_text": step2,
        "tokens": step3,
        "filtered_tokens": step4,
        "stemmed_tokens": step5,
        "final_text": final_text,
    }


def quick_pipeline(raw_text: str, extra_slang: dict = None):
    """Versi ringkas: hanya mengembalikan (clean_text, normalized_text, final_text)."""
    result = full_pipeline(raw_text, extra_slang=extra_slang)
    return result["clean_text"], result["normalized_text"], result["final_text"]
