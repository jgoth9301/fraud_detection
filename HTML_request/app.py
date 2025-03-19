import threading
import webbrowser
import os
import subprocess
import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from mlflow.tracking import MlflowClient

# ------------------------------
# App and Database Configuration
# ------------------------------

app = Flask(__name__, template_folder="templates")
app.secret_key = "supersecretkey"

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "instance", "fraud_detection.db")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + db_path
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------------------
# Flask-Login Setup
# ------------------------------

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ------------------------------
# Start MLflow Server
# ------------------------------

def start_mlflow_server():
    try:
        subprocess.Popen(
            [
                "mlflow", "server",
                "--backend-store-uri", "sqlite:///mlflow.db",
                "--default-artifact-root", os.path.join(current_dir, "mlruns"),
                "--host", "127.0.0.1",
                "--port", "5000"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("🚀 MLflow Server started!")
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
    print(f"📦 Loading model '{model_name}' version {newest_version.version}")
    return mlflow.pyfunc.load_model(model_uri)

model_name = "fraud_detection_rf"
try:
    model = load_newest_production_version(model_name)
    print(f"✅ Successfully loaded model: {model_name}")
except Exception as e:
    raise RuntimeError(f"🚧 Could not load model '{model_name}' from Model Registry: {e}")

# ------------------------------
# SQLAlchemy Models
# ------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="user")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def role_required(role):
    """Decorator to restrict certain routes to specific roles."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                abort(403)  # Forbidden
            return f(*args, **kwargs)
        return decorated_function
    return decorator

class FraudDetection(db.Model):
    __tablename__ = 'fraud_detection'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100))  # Optional
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

class ClaimTracking(db.Model):
    __tablename__ = 'claim_tracking'
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, nullable=False)
    claim_sum = db.Column(db.Float, nullable=False)

# ------------------------------
# Routes and Views
# ------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Root path. We allow both GET and POST here to avoid 405 errors
    if the form accidentally posts to "/".
    """
    return redirect(url_for("user_request"))

@app.route("/user_request", methods=["GET", "POST"])
def user_request():
    """Show the main user form (user_request.html)."""
    fraud_probability = None
    form_data = {}
    message = ""
    risk_color = "green"

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
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
            fraud_probability = float(model.predict(features)[0])

            if fraud_probability >= 4:
                message = "Your application is considered high risk. Please contact office@applicationform.com."
                risk_color = "red"
            else:
                message = "Your application has been received successfully!"
                risk_color = "green"

            # Optional: Save data in the FraudDetection table
            new_prediction = FraudDetection(
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
                risk_group=int(fraud_probability)
            )
            db.session.add(new_prediction)
            db.session.commit()
            print("💾 Prediction saved in fraud_detection table")

        except Exception as e:
            message = f"Error processing data: {e}"

    return render_template(
        "user_request.html",
        fraud_probability=fraud_probability or 0,
        form_data=form_data,
        message=message,
        fraud_risk_color=risk_color
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    """Show login.html."""
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
    """Log out the current user."""
    logout_user()
    flash("👋 Successfully logged out!", "info")
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    """Show register.html."""
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

@app.route("/dashboard")
@login_required
def dashboard():
    """Show dashboard.html."""
    return render_template("dashboard.html")

@app.route("/view_predictions")
@login_required
@role_required("admin")  # Only admin can access
def view_predictions():
    predictions = FraudDetection.query.all()  # Fetch all stored fraud detection results
    return render_template("predictions.html", predictions=predictions)

@app.route("/admin")
@login_required
@role_required("admin")
def admin_dashboard():
    """An admin-only route."""
    return "🔒 Admin Dashboard: Welcome, Admin!"

@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Endpoint to shut down the server (optional)."""
    print("Shutting down server...")
    os._exit(0)

# ------------------------------
# Start the Server and Create Tables
# ------------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        # Create an admin user if none exists
        if not User.query.filter_by(username="admin").first():
            admin_user = User(
                username="admin",
                password=generate_password_hash("admin123", method="pbkdf2:sha256"),
                role="admin"
            )
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Admin user created: Username='admin', Password='admin123'")

    print("🌍 Starting Flask server at http://127.0.0.1:8000/")
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        # Automatically open a browser tab to the home (which redirects to /user_request)
        threading.Timer(1.5, lambda: webbrowser.open_new("http://127.0.0.1:8000/")).start()

    app.run(debug=True, use_reloader=False, threaded=True, port=8000)
