import threading
import webbrowser
import os
import subprocess
import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from mlflow.tracking import MlflowClient

# Start MLflow Server, falls er nicht läuft
def start_mlflow_server():
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

# MLflow-Server starten
start_mlflow_server()

# MLflow Tracking-URI setzen
mlflow.set_tracking_uri("http://127.0.0.1:5000")

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

# Aktualisierter Modellname, passend zu model_training.py
model_name = "fraud_detection_rf"
try:
    model = load_newest_production_version(model_name)
    print(f"✅ Successfully loaded model: {model_name}")
except Exception as e:
    raise RuntimeError(f"🚧 Could not load model '{model_name}' from Model Registry: {e}")

# Flask App Setup
app = Flask(__name__, template_folder="templates")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///C:/Users/juerg/PycharmProjects/fraud_detection/HTML_request/instance/fraud_detection.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# SQLAlchemy Modell definieren
class fraud_detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100))
    customer_id = db.Column(db.String(100))
    age = db.Column(db.Float)
    gender = db.Column(db.Integer)
    marital_status = db.Column(db.Integer)
    occupation = db.Column(db.Integer)
    income_level = db.Column(db.Float)
    education_level = db.Column(db.Integer)
    credit_score = db.Column(db.Float)
    driving_record = db.Column(db.Integer)
    life_events = db.Column(db.Integer)
    risk_group = db.Column(db.Integer)

with app.app_context():
    db.create_all()

@app.route("/", methods=["GET", "POST"])
def home():
    fraud_probability = None
    form_data = {}
    message = ""
    risk_color = "green"

    if request.method == "POST":
        form_data = request.form.to_dict()
        print(f"🔍 Received form data: {form_data}")

        # Erwartete Felder (alle in Kleinbuchstaben, passend zum HTML)
        required_fields = ["age", "gender", "marital_status", "occupation",
                           "income_level", "education_level", "credit_score", "driving_record", "life_events"]

        for field in required_fields:
            if field not in form_data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        try:
            # Features-DataFrame mit korrekten Spaltennamen erstellen
            features = pd.DataFrame([{
                "age": float(form_data["age"]),
                "gender": form_data["gender"],
                "marital_status": form_data["marital_status"],
                "occupation": form_data["occupation"],
                "income_level": float(form_data["income_level"]),
                "education_level": form_data["education_level"],
                "credit_score": float(form_data["credit_score"]),
                "driving_record": form_data["driving_record"],
                "life_events": form_data["life_events"]
            }])

            # Encoding für kategoriale Variablen, um den Trainingsdaten zu entsprechen
            encoding_maps = {
                "gender": {"0": 0, "1": 1},
                "marital_status": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
                "education_level": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
                "occupation": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8},
                "driving_record": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
                "life_events": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
            }

            for col, mapping in encoding_maps.items():
                if col in features:
                    features[col] = features[col].astype(str).map(mapping)

            features.fillna(0, inplace=True)
            features = features.astype("float64")
            print(f"📝 Processed Features:\n{features}")

            # Vorhersage der Risk Group mittels des MLflow-Modells
            risk_group = int(model.predict(features)[0])
            fraud_probability = risk_group  # für die Ausgabe im Template

            # Loggen der Vorhersage in MLflow
            with mlflow.start_run():
                mlflow.log_param("model_used", model_name)
                mlflow.log_metric("risk_group", risk_group)

            # Messaging basierend auf der Risk Group
            if risk_group >= 4:
                message = "Your application is considered high risk. Please contact office@applicationform.com."
                risk_color = "red"
            else:
                message = "Your application has been received successfully!"
                risk_color = "green"

            # Persistierung der Daten in der Datenbank
            new_entry = fraud_detection(
                timestamp=form_data.get("timestamp"),
                customer_id=form_data.get("customer_id"),
                age=float(form_data["age"]),
                gender=int(form_data["gender"]),
                marital_status=int(form_data["marital_status"]),
                occupation=int(form_data["occupation"]),
                income_level=float(form_data["income_level"]),
                education_level=int(form_data["education_level"]),
                credit_score=float(form_data["credit_score"]),
                driving_record=int(form_data["driving_record"]),
                life_events=int(form_data["life_events"]),
                risk_group=risk_group
            )
            db.session.add(new_entry)
            db.session.commit()
            print("💾 Data persisted in fraud_detection.db")

        except Exception as e:
            return jsonify({"error": f"Data processing error: {e}"}), 500

    return render_template(
        "User_request.html",
        fraud_probability=fraud_probability,
        form_data=form_data,
        message=message,
        fraud_risk_color=risk_color
    )

@app.route("/shutdown", methods=["POST"])
def shutdown():
    print("Shutting down server...")
    os._exit(0)

if __name__ == "__main__":
    print("🌍 Starting Flask server at http://127.0.0.1:8000/")
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.5, lambda: webbrowser.open_new("http://127.0.0.1:8000/")).start()
    app.run(debug=True, use_reloader=False, threaded=True, port=8000)
