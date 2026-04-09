import sqlite3

DB_PATH = "chatbot.db"

def init_titles_table():
    """Create the chat_titles table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_chat_title(thread_id: str, title: str):
    """Insert or replace a chat title for a given thread."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO chat_titles (thread_id, title) VALUES (?, ?)",
        (str(thread_id), title)
    )
    conn.commit()
    conn.close()

def load_all_chat_titles() -> dict:
    """Return a dict of {thread_id: title} for all saved threads."""
    init_titles_table()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT thread_id, title FROM chat_titles").fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}