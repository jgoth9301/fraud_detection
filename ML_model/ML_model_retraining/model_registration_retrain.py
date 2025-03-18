import time
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Set the tracking URI to use your local mlruns directory.
mlflow.set_tracking_uri("file:///C:/Users/juerg/PycharmProjects/fraud_detection/ML_model/mlruns")


def register_best_model(
        csv_path,
        model_name="fraud_detection_rf",  # Name for your model in the registry
        artifact_path="final_rf_model_5classes_retrained"
        # Must match the artifact path used in mlflow.sklearn.log_model
):
    """
    Reads the CSV file with hyperparameter tuning results,
    finds the best model row (using the 'Result' column if available, or based on maximum accuracy otherwise),
    extracts the run_id, and registers the corresponding model in the MLflow Model Registry under 'model_name'.
    After registration, the model version is automatically promoted to 'Production'.
    """
    # 1) Read the CSV file with tuning results
    df = pd.read_csv(csv_path)

    # 2) Determine the best model row
    if "Result" in df.columns:
        best_rows = df.loc[df["Result"] == "YES"]
        if best_rows.empty:
            raise ValueError("No row with Result == 'YES' found in the CSV.")
        best_model_row = best_rows.iloc[0]
    else:
        print("Warning: Column 'Result' not found in CSV. Selecting best model based on maximum accuracy.")
        best_idx = df["accuracy"].idxmax()
        best_model_row = df.loc[best_idx]

    # 3) Extract the run_id for the best model
    best_run_id = best_model_row["run_id"]

    # 4) Construct the model URI using run_id and artifact_path
    model_uri = f"runs:/{best_run_id}/{artifact_path}"

    # 5) Register the model in the MLflow Model Registry
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    new_version = result.version
    print(f"✅ Model successfully registered!")
    print(f"   Name: {result.name}")
    print(f"   Version: {new_version}")

    # 6) Wait until the new model version reaches "READY" status (up to ~10 seconds)
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
            name=model_name,
            version=new_version
        )
        if model_version_details.status == "READY":
            break
        time.sleep(1)
    else:
        raise MlflowException(
            f"Model version {new_version} for {model_name} did not reach 'READY' status in time."
        )

    # 7) Promote the new model version to Production
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Production"
    )
    print(f"🚀 Model '{model_name}' (version {new_version}) has been promoted to Production.")


if __name__ == "__main__":
    # Full path to your hyperparameter tuning results CSV file
    tuning_results_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\ML_model\hyperparameter_results\hyperparameter_tuning_results_retrained.csv"

    register_best_model(
        csv_path=tuning_results_path,
        model_name="fraud_detection_rf",
        artifact_path="final_rf_model_5classes_retrained"
    )
