"""
================================================================================
MODULE: database.py
Lapisan akses data (Data Access Layer) untuk aplikasi.
Menyediakan operasi CRUD penuh (Create, Read, Update, Delete) di atas SQLite,
sehingga seluruh dataset komentar YouTube dapat dikelola langsung dari
antarmuka Streamlit tanpa perlu mengedit file Excel secara manual.
================================================================================
"""

import os
import sqlite3
import datetime as dt
from contextlib import contextmanager

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")
EXCEL_SEED_PATH = os.path.join(BASE_DIR, "data", "demo-comments.xlsx")

TABLE = "comments"

COLUMNS = [
    "aplikasi", "video_id", "judul_video", "channel", "tanggal_video",
    "keyword_sumber", "comment_id", "komentar_asli", "author",
    "tanggal_komentar", "like_count", "total_reply_count",
    "clean_text", "normalized_text", "final_text", "label",
    "is_preprocessed", "created_at", "updated_at",
]


# ------------------------------------------------------------------ #
# KONEKSI
# ------------------------------------------------------------------ #
@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------ #
# INISIALISASI & SEED
# ------------------------------------------------------------------ #
def init_db():
    """Membuat tabel jika belum ada."""
    with get_connection() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                aplikasi            TEXT,
                video_id            TEXT,
                judul_video         TEXT,
                channel             TEXT,
                tanggal_video       TEXT,
                keyword_sumber      TEXT,
                comment_id          TEXT,
                komentar_asli       TEXT NOT NULL,
                author              TEXT,
                tanggal_komentar    TEXT,
                like_count          INTEGER DEFAULT 0,
                total_reply_count   INTEGER DEFAULT 0,
                clean_text          TEXT,
                normalized_text     TEXT,
                final_text          TEXT,
                label               INTEGER,
                is_preprocessed     INTEGER DEFAULT 0,
                created_at          TEXT,
                updated_at          TEXT
            )
        """)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_aplikasi ON {TABLE}(aplikasi)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_label ON {TABLE}(label)")


def is_seeded() -> bool:
    with get_connection() as conn:
        cur = conn.execute(f"SELECT COUNT(*) AS n FROM {TABLE}")
        return cur.fetchone()["n"] > 0


def seed_from_excel(path: str = EXCEL_SEED_PATH, force: bool = False):
    """Mengisi database dari file Excel sumber. Jika force=True, data lama dihapus dulu."""
    init_db()
    if force:
        with get_connection() as conn:
            conn.execute(f"DELETE FROM {TABLE}")

    if not force and is_seeded():
        return 0

    df = pd.read_excel(path)
    # df = df.drop_duplicates(subset="comment_id")
    # df = df.drop_duplicates(subset="komentar_asli")
    # df = df[df["komentar_asli"].notna()]
    # df = df[df["komentar_asli"].astype(str).str.len() > 3]

    # TAMBAHKAN INI: Ganti baris kosong dengan string kosong agar tidak error NOT NULL di SQLite
    df["komentar_asli"] = df["komentar_asli"].fillna("")
    # ==========================================================================

    now = dt.datetime.now().isoformat(timespec="seconds")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            r.get("aplikasi"), r.get("video_id"), r.get("judul_video"),
            r.get("channel"), str(r.get("tanggal_video")), r.get("keyword_sumber"),
            r.get("comment_id"), str(r.get("komentar_asli")), r.get("author"),
            str(r.get("tanggal_komentar")), int(r.get("like_count") or 0),
            int(r.get("total_reply_count") or 0),
            None, None, None, None, 0, now, now,
        ))

    with get_connection() as conn:
        conn.executemany(f"""
            INSERT INTO {TABLE} (
                aplikasi, video_id, judul_video, channel, tanggal_video,
                keyword_sumber, comment_id, komentar_asli, author,
                tanggal_komentar, like_count, total_reply_count,
                clean_text, normalized_text, final_text, label,
                is_preprocessed, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

    return len(rows)


def reset_database():
    """Menghapus seluruh data dan mengisi ulang dari Excel sumber (hard reset)."""
    return seed_from_excel(force=True)


# ------------------------------------------------------------------ #
# READ
# ------------------------------------------------------------------ #
def fetch_all(search: str = None, aplikasi: str = None, label=None,
              preprocessed_only: bool = False, limit: int = None,
              offset: int = 0, order_by: str = "id DESC") -> pd.DataFrame:
    query = f"SELECT * FROM {TABLE} WHERE 1=1"
    params = []

    if search:
        query += " AND (komentar_asli LIKE ? OR author LIKE ? OR judul_video LIKE ?)"
        like = f"%{search}%"
        params += [like, like, like]

    if aplikasi and aplikasi != "Semua":
        query += " AND aplikasi = ?"
        params.append(aplikasi)

    if label is not None and label != "Semua":
        query += " AND label = ?"
        params.append(label)

    if preprocessed_only:
        query += " AND is_preprocessed = 1"

    query += f" ORDER BY {order_by}"

    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params += [limit, offset]

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def count_rows(search: str = None, aplikasi: str = None, label=None,
               preprocessed_only: bool = False) -> int:
    df = fetch_all(search=search, aplikasi=aplikasi, label=label,
                    preprocessed_only=preprocessed_only)
    return len(df)


def fetch_by_id(row_id: int) -> dict:
    with get_connection() as conn:
        cur = conn.execute(f"SELECT * FROM {TABLE} WHERE id = ?", (row_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def distinct_values(column: str) -> list:
    with get_connection() as conn:
        cur = conn.execute(f"SELECT DISTINCT {column} FROM {TABLE} WHERE {column} IS NOT NULL ORDER BY {column}")
        return [r[0] for r in cur.fetchall()]


def get_stats() -> dict:
    with get_connection() as conn:
        total = conn.execute(f"SELECT COUNT(*) c FROM {TABLE}").fetchone()["c"]
        labeled = conn.execute(f"SELECT COUNT(*) c FROM {TABLE} WHERE label IS NOT NULL").fetchone()["c"]
        preprocessed = conn.execute(f"SELECT COUNT(*) c FROM {TABLE} WHERE is_preprocessed = 1").fetchone()["c"]
        positif = conn.execute(f"SELECT COUNT(*) c FROM {TABLE} WHERE label = 1").fetchone()["c"]
        negatif = conn.execute(f"SELECT COUNT(*) c FROM {TABLE} WHERE label = 0").fetchone()["c"]
        n_apps = conn.execute(f"SELECT COUNT(DISTINCT aplikasi) c FROM {TABLE}").fetchone()["c"]
    return {
        "total": total, "labeled": labeled, "preprocessed": preprocessed,
        "positif": positif, "negatif": negatif, "n_apps": n_apps,
    }


# ------------------------------------------------------------------ #
# CREATE
# ------------------------------------------------------------------ #
def insert_comment(data: dict) -> int:
    now = dt.datetime.now().isoformat(timespec="seconds")
    data = {**data}
    data.setdefault("comment_id", f"manual-{int(dt.datetime.now().timestamp() * 1000)}")
    data.setdefault("like_count", 0)
    data.setdefault("total_reply_count", 0)
    data["created_at"] = now
    data["updated_at"] = now
    data["is_preprocessed"] = 0
    data["label"] = None
    data["clean_text"] = None
    data["normalized_text"] = None
    data["final_text"] = None

    fields = [c for c in COLUMNS if c in data]
    placeholders = ",".join(["?"] * len(fields))
    values = [data[f] for f in fields]

    with get_connection() as conn:
        cur = conn.execute(
            f"INSERT INTO {TABLE} ({','.join(fields)}) VALUES ({placeholders})", values
        )
        return cur.lastrowid


# ------------------------------------------------------------------ #
# UPDATE
# ------------------------------------------------------------------ #
def update_comment(row_id: int, data: dict):
    data = {**data}
    data["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    # jika komentar_asli diubah, tandai perlu preprocessing ulang
    if "komentar_asli" in data:
        data["is_preprocessed"] = 0
        data["clean_text"] = None
        data["normalized_text"] = None
        data["final_text"] = None
        data["label"] = None

    fields = [c for c in data if c in COLUMNS or c == "updated_at"]
    set_clause = ",".join([f"{f} = ?" for f in fields])
    values = [data[f] for f in fields] + [row_id]

    with get_connection() as conn:
        conn.execute(f"UPDATE {TABLE} SET {set_clause} WHERE id = ?", values)


def update_preprocessing_row(row_id: int, clean_text: str, normalized_text: str, final_text: str):
    now = dt.datetime.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute(f"""
            UPDATE {TABLE}
            SET clean_text = ?, normalized_text = ?, final_text = ?,
                is_preprocessed = 1, updated_at = ?
            WHERE id = ?
        """, (clean_text, normalized_text, final_text, now, row_id))


def bulk_update_preprocessing(rows: list):
    """rows: list of tuples (clean_text, normalized_text, final_text, id)"""
    now = dt.datetime.now().isoformat(timespec="seconds")
    rows = [(c, n, f, now, i) for (c, n, f, i) in rows]
    with get_connection() as conn:
        conn.executemany(f"""
            UPDATE {TABLE}
            SET clean_text = ?, normalized_text = ?, final_text = ?,
                is_preprocessed = 1, updated_at = ?
            WHERE id = ?
        """, rows)


def bulk_update_labels(rows: list):
    """rows: list of tuples (label, id)"""
    now = dt.datetime.now().isoformat(timespec="seconds")
    rows = [(lbl, now, i) for (lbl, i) in rows]
    with get_connection() as conn:
        conn.executemany(f"UPDATE {TABLE} SET label = ?, updated_at = ? WHERE id = ?", rows)


# ------------------------------------------------------------------ #
# DELETE
# ------------------------------------------------------------------ #
def delete_comment(row_id: int):
    with get_connection() as conn:
        conn.execute(f"DELETE FROM {TABLE} WHERE id = ?", (row_id,))


def delete_many(ids: list):
    if not ids:
        return
    placeholders = ",".join(["?"] * len(ids))
    with get_connection() as conn:
        conn.execute(f"DELETE FROM {TABLE} WHERE id IN ({placeholders})", ids)


def delete_all_comments() -> int:
    """Menghapus SELURUH baris data pada tabel comments (tanpa reset ulang dari Excel).
    Mengembalikan jumlah baris yang terhapus."""
    with get_connection() as conn:
        n = conn.execute(f"SELECT COUNT(*) AS n FROM {TABLE}").fetchone()["n"]
        conn.execute(f"DELETE FROM {TABLE}")
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (TABLE,))
        except sqlite3.OperationalError:
            pass
        return n
