import datetime
import calendar
from dateutil.relativedelta import relativedelta


def update_training_timeframe(csv_path):
    # Get the current date and time
    now = datetime.datetime.now()

    # Find the first day of the current month (at 00:00:00)
    current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Calculate the first day of the previous month
    previous_month_start = current_month_start - relativedelta(months=1)

    # Determine the last day of the previous month using calendar.monthrange
    last_day = calendar.monthrange(previous_month_start.year, previous_month_start.month)[1]

    # Set the end of the previous month at 23:59:59
    previous_month_end = previous_month_start.replace(day=last_day, hour=23, minute=59, second=59)

    # Format the dates as "dd.mm.yyyy HH:MM:SS"
    start_date_str = previous_month_start.strftime("%d.%m.%Y %H:%M:%S")
    end_date_str = previous_month_end.strftime("%d.%m.%Y %H:%M:%S")

    # Write the header and updated record to the CSV file
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("id;start_date;end_date\n")
        f.write(f"1;{start_date_str};{end_date_str}\n")


if __name__ == "__main__":
    # Path to the CSV file (adjust as needed)
    csv_file_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\ML_model\ML_model_retraining\training_timeframe.csv"

    update_training_timeframe(csv_file_path)
    print(f"CSV file updated successfully: {csv_file_path}")
