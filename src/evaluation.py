#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
import sys
sys.path.insert(1, './')
from functions import generate_features

def load_model(filepath):
    """
    Load a model from a given file path.
    """
    model = joblib.load(filepath) # Load the model from the specified file path using joblib
    return model

def predict(model, X, degree=9): # All final models are of the 9th degree
    """
    Make predictions using the loaded model and polynomial features.
    """
    X_poly = generate_features(X, degree) # Generate polynomial features for the input data
    predictions = model.predict(X_poly) # Use the model to make predictions based on the polynomial features
    return predictions

# Loading the model (example for a country model, replace with your own model path)
# model_path = '../models/final_models/your_country_model.joblib'
model_path = '../models/final_models/Germany_optimal_model_700.joblib'
model = load_model(model_path)  # Loading the previously trained and saved model

# Load or create new unseen data for predictions
# Here we create a DataFrame as an example. Replace this with the loading of your actual unseen data.
unseen_data = pd.DataFrame({
    'Day': np.arange(801, 901)  # Simulating day numbers from 801 to 900 for prediction
})

# Making predictions using the loaded model on the unseen data
predictions = predict(model, unseen_data[['Day']])

# Output or save the predictions as required
# For demonstration, we are printing the predictions here
print(predictions)

# You can also save the predictions to a CSV file or another format as needed
# predictions.to_csv('path_to_save_predictions.csv', index=False)
