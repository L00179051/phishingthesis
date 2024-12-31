import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd

# Load pre-trained models (Decision Tree, TF-IDF, PCA)
model_path = "Email_DTModel2.h5"
tfidf_vectorizer_path = "TfidfVectorizer_Model2.h5"
pca_model_path = "PCA_Model2.h5"

# Load the models
try:
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    pca_model = joblib.load(pca_model_path)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    exit()

# FastAPI app setup
app = FastAPI()

# Define the input model for the POST request
class EmailItem(BaseModel):
    email_text: str

# Function to preprocess the email text and predict the result
def predict_email(email_text):
    try:
        # Preprocess and vectorize the email text
        input_email = [email_text]
        email_vectorized = tfidf_vectorizer.transform(input_email)

        # Reduce the dimensionality of the vectorized email using PCA
        email_reduced = pca_model.transform(email_vectorized)

        # Make prediction using the trained model
        prediction = model.predict(email_reduced)
        print(prediction)

        # Return result
        if prediction[0] == 1:
            return "Phishing"
        else:
            return "Safe"

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return "Error during prediction"

# FastAPI POST endpoint for prediction
@app.post("/predict")
def predict_email_post(item: EmailItem):
    try:
        prediction = predict_email(item.email_text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error during prediction: " + str(e))

# If this file is run directly, it will launch the Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host= "0.0.0.0", port=8000)
