import sqlite3

# Pfad zur Datenbank
DB_PATH = r"C:\Users\juerg\PycharmProjects\fraud_detection\HTML_request\instance\fraud_detection.db"


def clear_table(table_name):
    """ Löscht den Inhalt einer Tabelle, aber behält die Struktur """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Prüfen, ob die Tabelle existiert
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        if cursor.fetchone() is None:
            print(f"⚠️ Tabelle '{table_name}' existiert nicht.")
            return

        # Löschen aller Daten aus der Tabelle
        cursor.execute(f"DELETE FROM {table_name};")
        conn.commit()
        print(f"✅ Tabelle '{table_name}' wurde erfolgreich geleert.")

        # Auto-Increment-Zähler zurücksetzen, wenn vorhanden
        try:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name=?;", (table_name,))
            conn.commit()
        except sqlite3.Error:
            print(f"ℹ️ Kein Auto-Increment in '{table_name}', daher wurde sqlite_sequence nicht zurückgesetzt.")

    except sqlite3.Error as e:
        print(f"⚠️ Fehler beim Leeren der Tabelle '{table_name}': {e}")

    finally:
        conn.close()


# Tabellen leeren
clear_table("fraud_detection")
clear_table("claim_tracking")
