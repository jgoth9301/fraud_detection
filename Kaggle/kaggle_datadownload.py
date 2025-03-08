import os
import kaggle

# Ensure the Kaggle API environment is correctly set
os.environ['KAGGLE_CONFIG_DIR'] = "C:/Users/juerg/PycharmProjects/fraud_detection/Kaggle"

# Dataset identifier and save location
dataset = "ravalsmit/insurance-claims-and-policy-data"
save_path = "C:/Users/juerg/PycharmProjects/fraud_detection/kaggle/data"

# Download and unzip the dataset
kaggle.api.dataset_download_files(dataset, path=save_path, unzip=True)

print("Download successful! Files are saved in:", save_path)