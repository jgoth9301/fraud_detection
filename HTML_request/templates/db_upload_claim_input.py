import sqlite3
import csv


def upload_claim_data():
    """
    Reads 'training_claim_input.csv' (with header: id;customer_id;claim_sum)
    and inserts each row into the 'claim_tracking' table in 'fraud_detection.db'.
    Ignores the CSV's 'id' column, since the DB auto-generates its own ID.
    """
    # Paths to your DB and CSV
    db_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\HTML_request\instance\fraud_detection.db"
    csv_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\HTML_request\templates\training_claim_input.csv"

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Open the CSV, specifying delimiter=';'
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=';')

        for row in reader:
            # row will have keys: "id", "customer_id", "claim_sum"
            # We ignore row["id"] and only insert the other two columns:
            customer_id = row["customer_id"]
            claim_sum = row["claim_sum"]

            # Insert into claim_tracking (customer_id, claim_sum)
            cursor.execute(
                """
                INSERT INTO claim_tracking (customer_id, claim_sum)
                VALUES (?, ?)
                """,
                (customer_id, claim_sum)
            )

    # Commit and close
    conn.commit()
    conn.close()
    print("Data from 'training_claim_input.csv' uploaded successfully to 'claim_tracking'!")


if __name__ == "__main__":
    upload_claim_data()


