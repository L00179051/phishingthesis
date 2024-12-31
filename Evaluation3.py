import pandas as pd
import requests
from sklearn.metrics import confusion_matrix, classification_report

# Define the model APIs
MODEL_APIS = {
    "Model 1": "http://ec2-34-207-230-69.compute-1.amazonaws.com:8000/predict",
    "Model 2": "http://ec2-3-144-10-32.us-east-2.compute.amazonaws.com:8000/predict",
    "Model 3": "http://ec2-54-215-191-237.us-west-1.compute.amazonaws.com:8000/predict",
}

# Function to get predictions from a model
def get_prediction(api_url, email_text):
    payload = {"email_text": email_text}
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("prediction", "Unknown")
    except Exception as e:
        return f"Error: {e}"

# Read the CSV file
input_file = "Data3.csv"
data = pd.read_csv(input_file)

# Initialize the output DataFrame
output_data = data.copy()

# Add columns for model predictions
for model_name in MODEL_APIS.keys():
    output_data[f"Predicted Output of {model_name}"] = None

# Iterate through the data and get predictions
for index, row in data.iterrows():
    email_text = row["Email"]
    for model_name, api_url in MODEL_APIS.items():
        output_data.at[index, f"Predicted Output of {model_name}"] = get_prediction(api_url, email_text)

# Save the output to a new CSV file
output_file = "Predicted Output 3.csv"
output_data.to_csv(output_file, index=False)
print(f"Predicted outputs saved to {output_file}")

# Generate confusion matrix and classification report for each model
actual_output = data["Actual Output"].replace({"Safe": 0, "Phishing": 1})

for model_name in MODEL_APIS.keys():
    predicted_output = output_data[f"Predicted Output of {model_name}"].replace({"Safe": 0, "Phishing": 1})
    try:
        cm = confusion_matrix(actual_output, predicted_output)
        cr = classification_report(actual_output, predicted_output, target_names=["Safe", "Phishing"])
        
        print(f"\nConfusion Matrix for {model_name}:")
        print(cm)
        print(f"\nClassification Report for {model_name}:")
        print(cr)
    except Exception as e:
        print(f"Error generating metrics for {model_name}: {e}")

