import sqlite3
import csv

def upload_customer_with_risk_data():
    """
    Reads 'training_customer_with_risk.csv' (semicolon-delimited) and inserts each row
    into the 'fraud_detection' table in 'fraud_detection.db'.
    Ignores the CSV's 'id' column since the DB auto-generates its own 'id'.
    """

    # Paths to your DB and CSV
    db_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\HTML_request\instance\fraud_detection.db"
    csv_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\HTML_request\templates\training_customer_with_risk.csv"

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Open the CSV, specifying delimiter=';' since your file uses semicolons
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=',')

        for row in reader:
            # row will have keys:
            # "id", "timestamp", "customer_id", "age", "gender", "marital_status",
            # "occupation", "income_level", "education_level", "credit_score",
            # "driving_record", "life_events", "Risk_Group"

            # We IGNORE the CSV's 'id' because the DB auto-increments its own 'id'
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
            # CSV column is "Risk_Group", DB column is "risk_group"
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

    # Commit and close
    conn.commit()
    conn.close()
    print("Data from 'training_customer_with_risk.csv' uploaded successfully to 'fraud_detection'!")

if __name__ == "__main__":
    upload_customer_with_risk_data()
