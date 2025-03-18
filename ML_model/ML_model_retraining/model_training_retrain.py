import os
import sqlite3
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature

def create_risk_group(df, sum_col="claim_sum"):
    """
    Uses an existing 'claim_sum' column to assign a risk group (1 to 5) via pd.qcut,
    then drops 'claim_sum' so it will not be used as a feature.
    """
    df["Risk_Group"] = pd.qcut(df[sum_col], q=5, labels=False, duplicates="drop") + 1
    df.drop(columns=[sum_col], inplace=True)
    return df

def train_and_evaluate_model(X_train, y_train, X_test, y_test, max_depth, n_estimators):
    """
    Trains a RandomForestClassifier with the given hyperparameters and returns performance metrics.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return rf, acc, prec, rec, f1

def main():
    """
    Main function:
    1) Reads start_date and end_date from 'training_timeframe.csv' (dayfirst format).
    2) Reads data from 'fraud_detection.db': 'fraud_detection' (customer features) and
       'claim_tracking' (claim_sum) tables.
    3) Converts DB 'timestamp' to datetime (dayfirst=True).
    4) Filters 'fraud_detection' by timestamp between start_date and end_date.
    5) Merges them on 'customer_id' -> single DataFrame.
    6) Creates a 1–5 Risk_Group from 'claim_sum'.
    7) Performs hyperparameter tuning for a RandomForest, logs runs to MLflow,
       saves results, prints best model info, retrains the best model, and logs it.
    """

    # 1) Read the date range from training_timeframe.csv (semicolon-delimited)
    timeframe_csv = r"C:\Users\juerg\PycharmProjects\fraud_detection\ML_model\ML_model_retraining\training_timeframe.csv"
    df_timeframe = pd.read_csv(timeframe_csv, delimiter=';')

    # We assume exactly one row: columns "id", "start_date", "end_date"
    start_date_str = df_timeframe.loc[0, "start_date"]  # e.g. "01.01.2025 00:00:00"
    end_date_str = df_timeframe.loc[0, "end_date"]      # e.g. "31.01.2025 23:59:59"

    # Convert to datetime with dayfirst=True so "01.01.2025 00:00:00" is parsed as 1 Jan 2025
    start_date_dt = pd.to_datetime(start_date_str, dayfirst=True)
    end_date_dt = pd.to_datetime(end_date_str, dayfirst=True)

    # 2) MLflow config
    mlflow.set_tracking_uri("file:///C:/Users/juerg/PycharmProjects/fraud_detection/ML_model/mlruns")
    mlflow.set_experiment("fraud_detection_retrained")

    # Path to the SQLite DB
    db_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\HTML_request\instance\fraud_detection.db"

    # 3) Connect and load data
    conn = sqlite3.connect(db_path)
    df_customer = pd.read_sql_query("SELECT * FROM fraud_detection", conn)
    df_claim = pd.read_sql_query("SELECT * FROM claim_tracking", conn)
    conn.close()

    # Convert 'timestamp' column to datetime, also dayfirst=True
    df_customer['timestamp'] = pd.to_datetime(
        df_customer['timestamp'],
        dayfirst=True,
        errors='coerce'
    )

    # === DEBUG: After parsing timestamps ===
    print("=== DEBUG: After parsing timestamps ===")
    print("df_customer shape:", df_customer.shape)
    # Print a few sample rows (timestamp + customer_id)
    print(df_customer[['timestamp', 'customer_id']].head(10))

    # 4) Filter the customer data by the chosen date range
    df_customer = df_customer[
        (df_customer['timestamp'] >= start_date_dt) &
        (df_customer['timestamp'] <= end_date_dt)
    ]

    print("\n=== DEBUG: After date filtering ===")
    print("df_customer shape:", df_customer.shape)
    print(df_customer[['timestamp', 'customer_id']].head(10))

    # 5) Merge the data on 'customer_id'
    df_merged = pd.merge(df_customer, df_claim, on="customer_id", how="inner")
    print("\n=== DEBUG: After merging with claim_tracking ===")
    print("df_merged shape:", df_merged.shape)
    if not df_merged.empty:
        print(df_merged[['timestamp', 'customer_id', 'claim_sum']].head(10))
    else:
        print("No rows in df_merged to show.")

    # If there's no data after merging, raise an error
    if df_merged.empty:
        raise ValueError(
            "No data after filtering by date range or merging. "
            "Possible reasons:\n"
            "1) The date range excludes all rows.\n"
            "2) No matching customer_id in claim_tracking.\n"
            "3) Timestamp parse errors (NaT)."
        )

    # 6) Create 1–5 Risk Group from 'claim_sum'
    df_merged = create_risk_group(df_merged, sum_col="claim_sum")

    # Define input columns and target
    input_columns = [
        "age", "gender", "marital_status", "occupation", "income_level",
        "education_level", "credit_score", "driving_record", "life_events"
    ]
    target_column = "Risk_Group"

    X = df_merged[input_columns].copy()
    y = df_merged[target_column].copy()

    # 7) If there's no data left, raise an error before fitting
    if X.empty:
        raise ValueError("No data available for training after all steps. Check your filters/columns.")

    # Scale features (optional)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Hyperparameter search space
    max_depth_values = [5, 7, 10]
    n_estimators_values = [50, 100, 200]

    # List to store results
    results = []

    # Hyperparameter Tuning
    for max_depth in max_depth_values:
        for n_estimators in n_estimators_values:
            print(f"Testing Model with max_depth={max_depth}, n_estimators={n_estimators}...")
            model, acc, prec, rec, f1 = train_and_evaluate_model(
                X_train, y_train, X_test, y_test, max_depth, n_estimators
            )

            # Log params & metrics
            mlflow.log_param(f"max_depth_{max_depth}_estimators_{n_estimators}", True)
            mlflow.log_metric(f"accuracy_{max_depth}_{n_estimators}", acc)
            mlflow.log_metric(f"precision_{max_depth}_{n_estimators}", prec)
            mlflow.log_metric(f"recall_{max_depth}_{n_estimators}", rec)
            mlflow.log_metric(f"f1_score_{max_depth}_{n_estimators}", f1)

            # Save results
            results.append({
                "experiment_id": mlflow.active_run().info.experiment_id,
                "run_id": mlflow.active_run().info.run_id,
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            })

    print("\nHyperparameter Tuning Completed!")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    tuning_results_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\ML_model\hyperparameter_results\hyperparameter_tuning_results_retrained.csv"
    os.makedirs(os.path.dirname(tuning_results_path), exist_ok=True)
    results_df.to_csv(tuning_results_path, index=False)
    print(f"Results saved to: {tuning_results_path}")

    # Find best model by accuracy
    best_idx = results_df["accuracy"].idxmax()
    results_df["Result"] = "NO"
    results_df.loc[best_idx, "Result"] = "YES"
    best_model_info = results_df.loc[best_idx]
    print("\nBest Model Found:")
    print(best_model_info)

    # Retrain final model with best hyperparams
    best_max_depth = int(best_model_info["max_depth"])
    best_n_estimators = int(best_model_info["n_estimators"])
    final_model, acc_final, prec_final, rec_final, f1_final = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, best_max_depth, best_n_estimators
    )

    # Log final params & metrics
    mlflow.log_param("best_max_depth", best_max_depth)
    mlflow.log_param("best_n_estimators", best_n_estimators)
    mlflow.log_metric("best_accuracy", acc_final)
    mlflow.log_metric("best_precision", prec_final)
    mlflow.log_metric("best_recall", rec_final)
    mlflow.log_metric("best_f1_score", f1_final)

    # Create model signature & log final model
    signature = infer_signature(X_train, final_model.predict(X_train))
    mlflow.sklearn.log_model(
        final_model,
        "final_rf_model_5classes_retrained",
        signature=signature,
        input_example=X_test[:5]
    )

    print("\nFinal model trained and logged to MLflow!")
    print(f"Final Accuracy: {acc_final:.3f}, F1-Score: {f1_final:.3f}")
    mlflow.end_run()

if __name__ == "__main__":
    main()
