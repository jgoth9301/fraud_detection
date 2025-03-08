import sqlite3

DB_PATH = "C:/Users/juerg/PycharmProjects/fraud_detection/HTML_request/fraud_detection.db"
conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
conn.close()

print("Tabellen in der Datenbank:", tables)

