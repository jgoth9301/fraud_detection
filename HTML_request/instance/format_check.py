import sqlite3
import pandas as pd


def main():
    # Path to your SQLite database
    db_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\HTML_request\instance\fraud_detection.db"

    # Connect to the database and load the "fraud_detection" table
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM fraud_detection"
    df = pd.read_sql_query(query, conn)
    conn.close()

    print("Total rows in 'fraud_detection':", df.shape[0])

    # Print first 10 raw timestamp values
    print("\nFirst 10 raw 'timestamp' values:")
    print(df['timestamp'].head(10))

    # Attempt to parse the timestamps using dayfirst=True
    df['parsed_timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

    # Count and print how many timestamps failed to parse
    nat_count = df['parsed_timestamp'].isna().sum()
    print("\nNumber of rows with unparsed timestamps (NaT):", nat_count)

    if nat_count > 0:
        print("\nSample rows with unparsed timestamps:")
        print(df.loc[df['parsed_timestamp'].isna(), ['timestamp']].head(10))
    else:
        print("\nAll timestamps parsed successfully.")


if __name__ == "__main__":
    main()
