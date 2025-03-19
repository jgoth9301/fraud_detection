import os
import datetime
import calendar
import csv


def update_training_timeframe_csv():
    # Use the directory where this script resides as the base directory.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # The existing CSV file to update in-place (make sure it exists if you don't allow creation).
    file_name = "training_timeframe.csv"
    file_path = os.path.join(base_dir, file_name)

    # (Optional) Check that the file already exists if you strictly don't want to create it.
    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"{file_path} does not exist. Cannot update in place.")

    # Get today's date.
    today = datetime.date.today()
    current_year = today.year
    current_month = today.month

    # Calculate the last completed month (the "past month" relative to current).
    # If it's January, we move to December of the previous year.
    last_full_month = current_month - 1
    last_full_month_year = current_year
    if last_full_month == 0:
        last_full_month = 12
        last_full_month_year -= 1

    # Determine the last day of that last_full_month.
    days_in_month = calendar.monthrange(last_full_month_year, last_full_month)[1]

    # Build start/end date strings:
    # Start date: first day of last_full_month at 00:00:00
    start_date = f"01.{last_full_month:02d}.{last_full_month_year} 00:00:00"
    # End date: last day of last_full_month at 23:59:59
    end_date = f"{days_in_month:02d}.{last_full_month:02d}.{last_full_month_year} 23:59:59"

    # Overwrite the file with a single row for this timeframe.
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["id", "start_date", "end_date"])
        writer.writerow([1, start_date, end_date])

    print(f"Updated {file_path} with a single row for {start_date} to {end_date}.")


if __name__ == "__main__":
    update_training_timeframe_csv()
