import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature

def create_risk_group(df, sum_col="Claim_Sum"):
    """
    Uses an existing 'Claim_Sum' column to assign a risk group (1 to 5) via pd.qcut,
    then drops 'Claim_Sum' so it will not be used as a feature.
    """
    # 1) Create a 1–5 risk group from the existing 'Claim_Sum'
    df["Risk_Group"] = pd.qcut(df[sum_col], q=5, labels=False, duplicates="drop") + 1

    # 2) Drop the 'Claim_Sum' column, keeping only 'Risk_Group'
    df.drop(columns=[sum_col], inplace=True)

    return df

def load_data(csv_path):
    """
    Loads the prepared CSV into a DataFrame.
    """
    return pd.read_csv(csv_path)

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
    Main function: loads data, uses the existing 'Claim_Sum' column to create a 1–5 Risk_Group,
    performs hyperparameter tuning for a RandomForest, logs all runs to MLflow, saves the tuning
    results to a CSV, and prints the best model info.
    """
    # 1) Explizit die Tracking-URI setzen:
    mlflow.set_tracking_uri("file:///C:/Users/juerg/PycharmProjects/fraud_detection/ML_model/mlruns")

    # Pfad zur vorbereiteten CSV
    prepared_data_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\ML_model\data\prepared_data.csv"

    # Pfad zum Speichern der Hyperparameter-Ergebnisse
    tuning_results_path = r"C:\Users\juerg\PycharmProjects\fraud_detection\ML_model\hyperparameter_results\hyperparameter_tuning_results.csv"

    # Sicherstellen, dass der Zielordner existiert
    os.makedirs(os.path.dirname(tuning_results_path), exist_ok=True)

    # 2) MLflow-Konfiguration: Experiment festlegen
    mlflow.set_experiment("fraud_detection_risk_model_5_classes")

    with mlflow.start_run(run_name="RandomForest-RiskGroup-5classes") as run:
        # Experiment- und Run-Infos
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id

        # 1) Daten laden
        df = load_data(prepared_data_path)

        # 2) 1–5 Risk Group erzeugen
        df = create_risk_group(df, sum_col="Claim_Sum")

        # 3) Input- und Zielspalten definieren
        input_columns = [
            "Age", "Gender", "Marital Status", "Occupation", "Income Level",
            "Education Level", "Credit Score", "Driving Record", "Life Events"
        ]
        target_column = "Risk_Group"

        X = df[input_columns].copy()
        y = df[target_column].copy()

        # Optional: Features skalieren
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Daten splitten
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Suchraum für Hyperparameter
        max_depth_values = [5, 7, 10]
        n_estimators_values = [50, 100, 200]

        # Liste für alle Ergebnisse
        results = []

        # Alle Kombinationen durchprobieren
        for max_depth in max_depth_values:
            for n_estimators in n_estimators_values:
                print(f"Testing Model with max_depth={max_depth}, n_estimators={n_estimators}...")

                model, acc, prec, rec, f1 = train_and_evaluate_model(
                    X_train, y_train, X_test, y_test, max_depth, n_estimators
                )

                # Loggen von Parametern und Metriken
                mlflow.log_param(f"max_depth_{max_depth}_estimators_{n_estimators}", True)
                mlflow.log_metric(f"accuracy_{max_depth}_{n_estimators}", acc)
                mlflow.log_metric(f"precision_{max_depth}_{n_estimators}", prec)
                mlflow.log_metric(f"recall_{max_depth}_{n_estimators}", rec)
                mlflow.log_metric(f"f1_score_{max_depth}_{n_estimators}", f1)

                # Speichern der Ergebnisse (inkl. experiment_id, run_id)
                results.append({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1
                })

        print("\nHyperparameter Tuning Completed!")

        # Ergebnisse in DataFrame
        results_df = pd.DataFrame(results)

        # Bestes Modell nach Accuracy
        best_idx = results_df["accuracy"].idxmax()

        # Bestes Modell markieren
        results_df["Result"] = "NO"
        results_df.loc[best_idx, "Result"] = "YES"

        # Ergebnisse als CSV speichern
        results_df.to_csv(tuning_results_path, index=False)
        print(f"Results saved to: {tuning_results_path}")

        # Info zum besten Modell ausgeben
        best_model_info = results_df.loc[best_idx]
        print("\nBest Model Found:")
        print(best_model_info)

        # Beste Hyperparameter extrahieren
        best_max_depth = int(best_model_info["max_depth"])
        best_n_estimators = int(best_model_info["n_estimators"])

        # Finales Modell erneut trainieren
        final_model, acc_final, prec_final, rec_final, f1_final = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, best_max_depth, best_n_estimators
        )

        # Finalen Satz Parameter/Metriken loggen
        mlflow.log_param("best_max_depth", best_max_depth)
        mlflow.log_param("best_n_estimators", best_n_estimators)
        mlflow.log_metric("best_accuracy", acc_final)
        mlflow.log_metric("best_precision", prec_final)
        mlflow.log_metric("best_recall", rec_final)
        mlflow.log_metric("best_f1_score", f1_final)

        # Model Signature erstellen
        signature = infer_signature(X_train, final_model.predict(X_train))

        # Finales Modell loggen
        mlflow.sklearn.log_model(
            final_model,
            "final_rf_model_5classes",
            signature=signature,
            input_example=X_test[:5]
        )

        print("\nFinal model trained and logged to MLflow!")
        print(f"Final Accuracy: {acc_final:.3f}, F1-Score: {f1_final:.3f}")

        mlflow.end_run()

if __name__ == "__main__":
    main()
