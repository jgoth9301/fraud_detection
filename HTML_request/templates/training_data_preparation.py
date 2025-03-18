import mlflow
import mlflow.pyfunc
import pandas as pd
import subprocess
from mlflow.tracking import MlflowClient

def start_mlflow_server():
    # Falls der MLflow-Server nicht läuft, kann er hier gestartet werden.
    try:
        subprocess.Popen(
            [
                "mlflow", "server",
                "--backend-store-uri", "sqlite:///mlflow.db",
                "--default-artifact-root", "./mlruns",
                "--host", "127.0.0.1",
                "--port", "5000"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("🚀 MLflow Server started!")
    except Exception as e:
        print(f"⚠️ Error starting MLflow server: {e}")

def load_newest_production_version(model_name: str):
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in all_versions if v.current_stage == "Production"]
    if not prod_versions:
        raise ValueError(f"No versions of model '{model_name}' in Production stage.")
    newest_version = max(prod_versions, key=lambda v: v.last_updated_timestamp)
    model_uri = f"models:/{model_name}/{newest_version.version}"
    print(f"📦 Loading model '{model_name}' version {newest_version.version} (updated: {newest_version.last_updated_timestamp})")
    return mlflow.pyfunc.load_model(model_uri)

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wendet Encoding-Mappings auf die kategorialen Variablen an und konvertiert numerische Spalten.
    Erwartete Spalten: age, gender, marital_status, occupation, income_level,
    education_level, credit_score, driving_record, life_events.
    """
    encoding_maps = {
        "gender": {"0": 0, "1": 1},
        "marital_status": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
        "education_level": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
        "occupation": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8},
        "driving_record": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
        "life_events": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
    }

    for col, mapping in encoding_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping)

    # Konvertiere numerische Spalten
    for col in ["age", "income_level", "credit_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.fillna(0, inplace=True)
    # Stelle sicher, dass alle relevanten Spalten als float vorliegen
    for col in ["age", "gender", "marital_status", "occupation", "income_level", "education_level", "credit_score", "driving_record", "life_events"]:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    return df

def main():
    # Optionale Serverstart, falls nötig:
    # start_mlflow_server()  # Unkommentieren, wenn der MLflow-Server noch nicht läuft

    # MLflow Tracking-URI setzen
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Modell laden (verwende den gleichen Modellnamen wie im Training)
    model_name = "fraud_detection_rf"
    try:
        model = load_newest_production_version(model_name)
        print(f"✅ Successfully loaded model: {model_name}")
    except Exception as e:
        raise RuntimeError(f"🚧 Could not load model '{model_name}' from Model Registry: {e}")

    # CSV-Datei einlesen – passe den Dateipfad ggf. an
    input_file = "training_customer_input.csv"
    try:
        df = pd.read_csv("training_customer_input.csv", sep=";")
        print(f"📥 Loaded data from {input_file} with shape {df.shape}")
    except Exception as e:
        raise RuntimeError(f"Error reading {input_file}: {e}")

    # Überprüfe, ob alle erforderlichen Spalten vorhanden sind
    required_columns = ["age", "gender", "marital_status", "occupation",
                        "income_level", "education_level", "credit_score", "driving_record", "life_events"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Input data is missing required columns: {', '.join(missing_cols)}")

    # Features vorbereiten und vorverarbeiten
    features_df = df[required_columns].copy()
    features_df = preprocess_features(features_df)
    print("📝 Preprocessing complete. Sample:")
    print(features_df.head())

    # Vorhersage der Risk_Group
    predictions = model.predict(features_df)
    # Füge die Vorhersagen als neue Spalte hinzu
    df["Risk_Group"] = predictions.astype(int)
    print("✅ Risk_Group predictions added to the DataFrame.")

    # Speichere das aktualisierte DataFrame in eine neue CSV-Datei
    output_file = "training_customer_with_risk.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Updated file saved as {output_file}")

if __name__ == "__main__":
    main()
