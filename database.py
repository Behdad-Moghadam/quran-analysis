import sqlite3
from enum import Enum

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('farsi_arabic_database.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS Farsi_arabic_method1 (
        id INTEGER PRIMARY KEY,
        farsi_text TEXT UNIQUE,
        arabic_texts TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS Farsi_english_method1 (
        id INTEGER PRIMARY KEY,
        farsi_text TEXT UNIQUE,
        english_texts TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS Farsi_arabic_method2 (
        id INTEGER PRIMARY KEY,
        farsi_text TEXT UNIQUE,
        arabic_texts TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS Farsi_english_method2 (
        id INTEGER PRIMARY KEY,
        farsi_text TEXT UNIQUE,
        english_texts TEXT
    )
""")

class Method(Enum):
    ONE = "method1"
    TWO = "method2"

def add_farsi_arabic_text(farsi_text, arabic_texts, method: Method):
    try:
        cursor.execute(f"INSERT INTO Farsi_arabic_{method.value} (farsi_text, arabic_texts) VALUES (?, ?)", (farsi_text, arabic_texts))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Farsi Text, '{farsi_text}' ,already exists in Farsi_arabic_{method.value} table.")

def add_farsi_english_text(farsi_text, english_texts, method: Method):
    try:
        cursor.execute(f"INSERT INTO Farsi_english_{method.value} (farsi_text, english_texts) VALUES (?, ?)", (farsi_text, english_texts))
    except sqlite3.IntegrityError:
        print(f"Farsi Text, '{farsi_text}' ,already exists in Farsi_english_{method.value} table.")

def get_arabic_texts(farsi_text, method: Method|str):
    method_name = method if type(method) is str else method.value 
    cursor.execute(f"SELECT arabic_texts FROM Farsi_arabic_{method_name} WHERE farsi_text = ?", (farsi_text,))
    return cursor.fetchall()

def get_english_texts(farsi_text, method: Method|str):
    method_name = method if type(method) is str else method.value 
    cursor.execute(f"SELECT english_texts FROM Farsi_english_{method_name} WHERE farsi_text = ?", (farsi_text,))
    return cursor.fetchall()


# conn.close()   # Is it necessary?

