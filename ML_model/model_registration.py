import time
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Explizit die Tracking-URI setzen, sodass das mlruns-Verzeichnis verwendet wird.
mlflow.set_tracking_uri("file:///C:/Users/juerg/PycharmProjects/fraud_detection/ML_model/mlruns")

def register_best_model(
        csv_path,
        model_name="fraud_detection_rf",  # Geben Sie Ihrem Modell einen Namen
        artifact_path="final_rf_model_5classes"  # Pfad, der in mlflow.sklearn.log_model verwendet wurde
):
    """
    Liest die CSV-Datei mit den Ergebnissen der Hyperparameter-Tuning,
    sucht die Zeile, in der 'Result' == 'YES' steht, extrahiert die run_id
    und registriert das Modell in der MLflow Model Registry unter 'model_name'.
    Nach der Registrierung wird die Modellversion automatisch auf 'Production'
    gesetzt.
    """
    # 1) CSV-Datei einlesen
    df = pd.read_csv(csv_path)

    # 2) Zeile mit dem besten Modell finden (Result == "YES")
    best_model_row = df.loc[df["Result"] == "YES"].iloc[0]

    # 3) run_id extrahieren
    best_run_id = best_model_row["run_id"]

    # 4) Modell-URI basierend auf run_id und artifact_path konstruieren
    model_uri = f"runs:/{best_run_id}/{artifact_path}"

    # 5) Modell in der MLflow Model Registry registrieren
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    new_version = result.version
    print(f"✅ Model successfully registered!")
    print(f"   Name: {result.name}")
    print(f"   Version: {new_version}")

    # 6) Warten, bis der Modellstatus auf "READY" steht
    client = MlflowClient()
    for _ in range(10):  # bis zu ca. 10 Sekunden
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

    # 7) Die neu registrierte Modellversion in Production befördern
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Production"
    )
    print(f"🚀 Model '{model_name}' (version {new_version}) has been promoted to Production.")

if __name__ == "__main__":
    # Pfad zur CSV-Datei mit den Hyperparameter-Tuning-Ergebnissen
    tuning_results_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\ML_model\hyperparameter_results\hyperparameter_tuning_results.csv"

    # Funktion aufrufen, um das beste Modell zu registrieren und in Production zu setzen
    register_best_model(
        csv_path=tuning_results_path,
        model_name="fraud_detection_rf",        # Wählen Sie einen aussagekräftigen Namen für das Registry-Modell
        artifact_path="final_rf_model_5classes"   # Der in mlflow.sklearn.log_model verwendete Artifact-Pfad
    )
