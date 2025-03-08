import sqlite3
import pandas as pd


def query_database(db_path, table_name):
    """
    Verbindet sich mit der SQLite-Datenbank und gibt die ersten 100 Zeilen aus der angegebenen Tabelle zurück.
    """
    try:
        # Verbindung zur SQLite-Datenbank herstellen
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Setze den Tabellennamen in Anführungszeichen, um Syntaxfehler zu vermeiden
        query = f'SELECT * FROM "{table_name}" LIMIT 100'
        df = pd.read_sql_query(query, conn)

        # Verbindung schließen
        conn.close()

        print(df.head(100))  # Anzeige der ersten 100 Zeilen in der Konsole

        return df
    except Exception as e:
        print(f"Fehler beim Abfragen der Datenbank: {e}")
        return None


# Beispielhafte Nutzung
db_path = "fraud_detection.db"  # Pfad zur Datenbank

table_name = "fraud_detection"  # Tabellennamen explizit definieren

# Anzahl der Zeilen in der Tabelle überprüfen
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
row_count = cursor.fetchone()[0]
conn.close()
print(f"🔍 Anzahl der Zeilen in der Tabelle '{table_name}': {row_count}")

# Falls du den Tabellennamen kennst, kannst du die Funktion so ausführen:
df = query_database(db_path, table_name)