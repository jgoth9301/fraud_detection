import sqlite3
import csv
import os


def upload_claim_data():
    """
    Reads 'training_claim_input.csv' (with header: id;customer_id;claim_sum)
    and inserts each row into the 'claim_tracking' table in 'fraud_detection.db'.
    Ignores the CSV's 'id' column, since the DB auto-generates its own ID.
    """
    # Dynamically determine the current script directory (templates folder).
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory of current_dir (i.e., HTML_request).
    parent_dir = os.path.dirname(current_dir)

    # Build the path to the database file within the 'instance' folder (inside HTML_request).
    db_path = os.path.join(parent_dir, "instance", "fraud_detection.db")

    # Build the path to the CSV file within the 'templates' folder.
    csv_path = os.path.join(current_dir, "training_claim_input.csv")

    # Debug: print the paths to verify they're correct
    print("Database path:", db_path)
    print("CSV path:", csv_path)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Open the CSV file with delimiter ';'
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            # The CSV has keys: "id", "customer_id", "claim_sum"
            # We ignore "id" and only insert "customer_id" and "claim_sum"
            customer_id = row["customer_id"]
            claim_sum = row["claim_sum"]

            # Insert the data into the claim_tracking table
            cursor.execute(
                """
                INSERT INTO claim_tracking (customer_id, claim_sum)
                VALUES (?, ?)
                """,
                (customer_id, claim_sum)
            )

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Data from 'training_claim_input.csv' uploaded successfully to 'claim_tracking'!")


if __name__ == "__main__":
    upload_claim_data()
