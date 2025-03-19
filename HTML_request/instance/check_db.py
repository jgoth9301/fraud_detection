import os
import sqlite3

# Dynamically determine the current script directory (which is already the 'instance' folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the database file located in the same folder
db_path = os.path.join(current_dir, "fraud_detection.db")

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Retrieve all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tabellen in der Datenbank:")
for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    print(f"📌 {table_name}: {row_count} Zeilen")

# Close the connection
conn.close()

