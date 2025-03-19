import os
import sqlite3
import csv


def upload_customer_with_risk_data():
    """
    Reads 'training_customer_with_risk.csv' (semicolon-delimited) and inserts each row
    into the 'fraud_detection' table in 'fraud_detection.db'.
    Ignores the CSV's 'id' column since the DB auto-generates its own 'id'.
    """
    # Dynamically determine the current script directory (assumed to be templates folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory of current_dir (which is HTML_request)
    parent_dir = os.path.dirname(current_dir)

    # Build the path to the database file within the 'instance' folder
    db_path = os.path.join(parent_dir, "instance", "fraud_detection.db")

    # Build the path to the CSV file within the 'templates' folder
    csv_path = os.path.join(current_dir, "training_customer_with_risk.csv")

    # Debugging: Print paths to verify correctness
    print("Database path:", db_path)
    print("CSV path:", csv_path)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Open the CSV, specifying delimiter=',' since your file uses semicolons
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            # row will have keys:
            # "id", "timestamp", "customer_id", "age", "gender", "marital_status",
            # "occupation", "income_level", "education_level", "credit_score",
            # "driving_record", "life_events", "Risk_Group"
            timestamp = row["timestamp"]
            customer_id = row["customer_id"]
            age = row["age"]
            gender = row["gender"]
            marital_status = row["marital_status"]
            occupation = row["occupation"]
            income_level = row["income_level"]
            education_level = row["education_level"]
            credit_score = row["credit_score"]
            driving_record = row["driving_record"]
            life_events = row["life_events"]
            risk_group = row["Risk_Group"]

            # Insert into fraud_detection (excluding the auto-generated 'id')
            cursor.execute(
                """
                INSERT INTO fraud_detection 
                (timestamp, customer_id, age, gender, marital_status, occupation,
                 income_level, education_level, credit_score, driving_record, life_events, risk_group)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, customer_id, age, gender, marital_status,
                    occupation, income_level, education_level, credit_score,
                    driving_record, life_events, risk_group
                )
            )

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Data from 'training_customer_with_risk.csv' uploaded successfully to 'fraud_detection'!")


if __name__ == "__main__":
    upload_customer_with_risk_data()
