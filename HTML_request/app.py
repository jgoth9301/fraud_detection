import threading
import webbrowser
import os
import subprocess
import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from mlflow.tracking import MlflowClient

# Initialize Flask App
app = Flask(__name__, template_folder="templates")
app.secret_key = "supersecretkey"

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fraud_detection.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# Start MLflow Server if not running
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
        print("\U0001F680 MLflow Server started!")
    except Exception as e:
        print(f"⚠️ Error starting MLflow server: {e}")


start_mlflow_server()
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def load_newest_production_version(model_name: str):
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in all_versions if v.current_stage == "Production"]
    if not prod_versions:
        raise ValueError(f"No versions of model '{model_name}' in Production stage.")
    newest_version = max(prod_versions, key=lambda v: v.last_updated_timestamp)
    model_uri = f"models:/{model_name}/{newest_version.version}"
    print(f"\U0001F4E6 Loading model '{model_name}' version {newest_version.version}")
    return mlflow.pyfunc.load_model(model_uri)


model_name = "fraud_detection_rf"
try:
    model = load_newest_production_version(model_name)
    print(f"✅ Successfully loaded model: {model_name}")
except Exception as e:
    raise RuntimeError(f"🚧 Could not load model '{model_name}' from Model Registry: {e}")


# User Model with Role-based Access Control
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="user")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                abort(403)
            return f(*args, **kwargs)

        return decorated_function

    return decorator


@app.route("/", methods=["GET", "POST"])
def home():
    fraud_probability = None  # Ensure fraud_probability is always defined
    form_data = {}
    message = ""
    risk_color = "green"

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            features = pd.DataFrame([{  # Convert form data to DataFrame
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

            # Ensure categorical values are properly encoded
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

            # Perform fraud risk prediction
            fraud_probability = float(model.predict(features)[0])  # Ensure fraud_probability is a number

            # Define risk levels
            if fraud_probability >= 4:
                message = "Your application is considered high risk. Please contact office@applicationform.com."
                risk_color = "red"
            else:
                message = "Your application has been received successfully!"
                risk_color = "green"

        except Exception as e:
            message = f"Error processing data: {e}"

    return render_template("User_request.html", fraud_probability=fraud_probability or 0, fraud_risk_color=risk_color, message=message)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        role = request.form.get("role", "user")

        if User.query.filter_by(username=username).first():
            flash("⚠️ Username already taken!", "warning")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(username=username, password=hashed_password, role=role)

        db.session.add(new_user)
        db.session.commit()
        flash("🎉 Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(f"✅ Welcome, {user.role}!", "success")
            return redirect(url_for("dashboard"))

        flash("⚠️ Invalid username or password!", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("👋 Successfully logged out!", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/view_predictions")
@login_required
@role_required("admin")  # Only admin can access
def view_predictions():
    predictions = FraudDetection.query.all()  # Fetch all stored fraud detection results
    return render_template("predictions.html", predictions=predictions)

@app.route("/user_request")
@login_required
def user_request():
    if current_user.role != "user":
        abort(403)  # Prevent non-users from accessing this page
    return render_template("User_request.html")

class FraudDetection(db.Model):
    __tablename__ = 'fraud_detection'  # This specifies the table name in the database
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    risk_group = db.Column(db.Integer, nullable=False)

@app.route("/admin")
@login_required
@role_required("admin")
def admin_dashboard():
    return "🔒 Admin Dashboard: Welcome, Admin!"


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        if not User.query.filter_by(username="admin").first():
            admin_user = User(username="admin", password=generate_password_hash("admin123", method="pbkdf2:sha256"),
                              role="admin")
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Admin user created: Username='admin', Password='admin123'")

    print("🌍 Starting Flask server at http://127.0.0.1:8000/")
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.5, lambda: webbrowser.open_new("http://127.0.0.1:8000/"))
    app.run(debug=True, use_reloader=False, threaded=True, port=8000)
