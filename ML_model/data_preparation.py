import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
data_path = "/Kaggle_Download/data/data_synthetic.csv"
df = pd.read_csv(data_path)
print("✅ Data successfully loaded!")

# Remove the Risk_Score column if it exists
if "Risk_Score" in df.columns:
    df.drop(columns=["Risk_Score"], inplace=True)
    print("✅ Risk_Score column removed.")

# 1️⃣ **Feature Engineering: Prepare the data**
# Define input and result columns
input_columns = [
    "Age", "Gender", "Marital Status", "Occupation", "Income Level", "Education Level", "Credit Score", "Driving Record","Life Events"
]
result_columns = ["Claim History", "Previous Claims History"]

# Remove rows with missing values in important columns
df.dropna(subset=result_columns, inplace=True)

# ➕ Create a new result column "Claim_Sum"
df["Claim_Sum"] = df["Claim History"] + df["Previous Claims History"]
result_columns.append("Claim_Sum")  # Add it to the result_columns list

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for future use

print("✅ Categorical values encoded.")

# 2️⃣ **Save prepared data**
# First save as prepared_data.csv (if needed)
output_prepared_path = "C:/Users/juerg/PycharmProjects/fraud_detection/ML_model/data/prepared_data.csv"
# Ensure directory exists for prepared_data.csv
output_dir = os.path.dirname(output_prepared_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
df.to_csv(output_prepared_path, index=False)

print(f"💾 Prepared data saved at: {output_prepared_path}")
