import time
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def register_best_model(
        csv_path,
        model_name="fraud_detection_rf",  # Name for your model in the registry
        artifact_path="final_rf_model_5classes_retrained"  # Must match the artifact path used in mlflow.sklearn.log_model
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
    print(f"Best run id: {best_run_id}")

    # 4) Check if the run exists in the MLflow tracking store
    client = MlflowClient()
    try:
        run_info = client.get_run(best_run_id)
    except Exception as e:
        raise MlflowException(
            f"Run '{best_run_id}' not found in MLflow tracking store. "
            f"Please ensure that the run exists and has not been deleted. Error: {e}"
        )

    # 5) Construct the model URI using run_id and artifact_path
    model_uri = f"runs:/{best_run_id}/{artifact_path}"
    print(f"Model URI: {model_uri}")

    # 6) Register the model in the MLflow Model Registry
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    new_version = result.version
    print(f"✅ Model successfully registered!")
    print(f"   Name: {result.name}")
    print(f"   Version: {new_version}")

    # 7) Wait until the new model version reaches "READY" status (up to ~10 seconds)
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

    # 8) Promote the new model version to Production
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Production"
    )
    print(f"🚀 Model '{model_name}' (version {new_version}) has been promoted to Production.")

if __name__ == "__main__":
    # If running in CI (e.g. GitHub Actions), use the repository root (os.getcwd())
    # and set a new experiment name to avoid cached Windows paths.
    if os.getenv("CI"):
        base_dir = os.getcwd()
        experiment_name = "fraud_detection_retrained_CI"
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        experiment_name = "fraud_detection_retrained"
    # Normalize base_dir to use forward slashes
    base_dir = base_dir.replace("\\", "/")
    print(f"Base directory: {base_dir}")

    # Set the MLflow tracking URI to the mlruns folder relative to the repository root.
    tracking_uri = "file:///" + os.path.join(base_dir, "ML_model", "mlruns")
    tracking_uri = tracking_uri.replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Using tracking URI: {mlflow.get_tracking_uri()}")

    # Set experiment name (this will create a new experiment in CI with a Linux-friendly artifact_uri)
    mlflow.set_experiment(experiment_name)
    print(f"Using experiment: {experiment_name}")

    # Build the full path to the hyperparameter tuning results CSV file.
    tuning_results_path = os.path.join(base_dir, "ML_model", "hyperparameter_results", "hyperparameter_tuning_results_retrained.csv")
    print(f"Using tuning results CSV file at: {tuning_results_path}")

    register_best_model(
        csv_path=tuning_results_path,
        model_name="fraud_detection_rf",
        artifact_path="final_rf_model_5classes_retrained"
    )

