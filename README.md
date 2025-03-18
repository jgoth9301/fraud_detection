# Fraud Detection in a Government Agency – MLOps

This repository implements a fraud detection solution designed for a government agency. The project integrates data acquisition, automated machine learning model retraining, and a web-based interface for result display and administration. The MLOps practices employed here help ensure that the model stays up-to-date with periodic retraining and registration.

## Repository Structure

Below is a description of every file and folder in the repository:

### Top-Level Files

- **.gitignore**  
  Specifies files and directories that should be ignored by Git (e.g., local configurations, logs, or other temporary files).

- **README.md**  
  This file – providing an overview of the project, descriptions of the repository contents, and instructions for installation, usage, and testing.

- **requirements.txt**  
  Contains a list of all Python dependencies required to run the project (for example, Flask, SQLAlchemy, pandas, python-dateutil, etc.). Run `pip install -r requirements.txt` to install them.

### .github Folder

- **.github/workflows/ci.yaml**  
  Contains the GitHub Actions workflow definition that automates the monthly model retraining process. The workflow:
  - Runs on the first day of each month (UTC midnight).
  - Updates the training timeframe CSV file with the previous month’s start and end dates.
  - Executes the model retraining script.
  - Executes the model registration script.

### HTML_request Folder

This folder contains the HTML templates for the web interface.

- **dashboard.html**  
  The main dashboard template that welcomes the user and displays different options based on their role. For example:
  - **Admins** see buttons for “Result Predictions” and “Admin Configuration.”
  - **Regular users** see an option to start a new fraud detection request.

- **predictions.html**  
  An admin-only view that displays fraud detection predictions in a table format. This template iterates over a collection of prediction records passed from the Flask view.

### Kaggle_Download Folder

This folder includes scripts and resources for downloading datasets from Kaggle. The scripts here automate the process of retrieving updated data, which is then used in the fraud detection analysis and model training.  
> *Note: Specific file names are not detailed, but expect one or more Python scripts that use the Kaggle API or similar functionality.*

### ML_model Folder

Houses all components related to the machine learning model.

#### ML_model/ML_model_retraining Subfolder

- **training_timeframe.csv**  
  A CSV file that holds the training timeframe information. It uses a semicolon-separated format with the header:  
  `id;start_date;end_date`  
  An example record might be:  
  `1;01.01.2025 00:00:00;31.12.2025 23:59:59`  
  This file is updated monthly by the GitHub Actions workflow to reflect the previous month’s timeframe.

- **model_training_retrain.py**  
  A Python script that retrains the fraud detection model using the updated timeframe from `training_timeframe.csv`. This script is executed automatically by the CI workflow.

- **model_registration_retrain.py**  
  A Python script that registers the newly trained model into the system. This ensures that the latest model is deployed for fraud detection operations. It is also executed by the CI workflow.

## Usage

1. **Automated Retraining:**  
   GitHub Actions will run the workflow (`ci.yaml`) on the first day of each month. The workflow updates the training timeframe, retrains the model, and registers it.

2. **Web Interface:**  
   The HTML templates (in the `HTML_request` folder) are used by a Flask application to render the dashboard and predictions views. The dashboard adapts its options based on whether the user is an admin or a regular user.

3. **Data Handling:**  
   Use the scripts in the `Kaggle_Download` folder to update or download new datasets from Kaggle as needed.

## Getting Started

- **Clone the repository:**

  ```bash
  git clone https://github.com/jgoth9301/fraud_detection.git
  cd fraud_detection

Install dependencies:
```shell
pip install -r requirements.txt
```

Testing the Workflows:
The GitHub Actions workflow can be monitored and, if needed, triggered manually from the repository’s Actions tab.

This README provides a concise overview of each component within the repository. For more detailed documentation, please refer to the inline comments in the scripts and templates.

---

You can now save this content as `README.md` in your repository for direct download and reference.
