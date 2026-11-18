"""
================================================================================
MODULE: lexicon.py
Pelabelan sentimen berbasis leksikon (lexicon-based), sesuai STEP 9 pada
notebook SKRIPSI.ipynb. Leksikon dasar disediakan di sini, namun dapat
diperluas secara interaktif dari halaman "Labeling Sentimen" di aplikasi
(disimpan pada st.session_state agar CRUD leksikon tidak memerlukan restart).
================================================================================
"""

POSITIVE_WORDS = {
    "bagus", "keren", "mantap", "baik", "terima", "kasih",
    "membantu", "berguna", "hebat", "luar", "biasa",
    "suka", "senang", "puas", "recommended", "worth",
    "cepat", "mudah", "akurat", "praktis", "lengkap",
    "memuaskan", "terbaik", "top", "excellent", "love",
}

NEGATIVE_WORDS = {
    "buruk", "jelek", "salah", "lambat", "lemot",
    "benci", "kesal", "gagal", "error", "rusak",
    "parah", "worst", "tidak", "tak", "bukan",
    "kurang", "mahal", "sulit", "susah", "bingung",
    "lamban", "payah", "kecewa", "halusinasi", "ngaco",
    "cacat", "bohong", "lambatnya", "buruknya",
}

NEGATION_WORDS = {"tidak", "tak", "bukan"}


def label_sentiment(tokens: list, positive_words=None, negative_words=None) -> int:
    """1 = Positif, 0 = Negatif. Sama persis dengan logika di notebook."""
    pos_words = positive_words if positive_words is not None else POSITIVE_WORDS
    neg_words = negative_words if negative_words is not None else NEGATIVE_WORDS

    pos_score = sum(1 for w in tokens if w in pos_words)
    neg_score = sum(1 for w in tokens if w in neg_words)

    if pos_score > neg_score:
        return 1
    elif neg_score > pos_score:
        return 0
    else:
        if any(w in NEGATION_WORDS for w in tokens):
            return 0
        return 1


def label_from_text(final_text: str, positive_words=None, negative_words=None) -> int:
    tokens = final_text.split() if final_text else []
    return label_sentiment(tokens, positive_words, negative_words)
