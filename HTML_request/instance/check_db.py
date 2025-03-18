import sqlite3

DB_PATH = "C:/Users/juerg/PycharmProjects/fraud_detection/HTML_request/instance/fraud_detection.db"

# Verbindung zur SQLite-Datenbank herstellen
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Alle Tabellennamen abrufen
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tabellen in der Datenbank:")
for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    print(f"📌 {table_name}: {row_count} Zeilen")

# Verbindung schließen
conn.close()
