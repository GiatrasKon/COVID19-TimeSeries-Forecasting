#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
import joblib
import os

def generate_features(df, degree):
    """
    Generate polynomial features.
    """
    poly = PolynomialFeatures(degree=degree) # Initializes PolynomialFeatures from sklearn with the given degree
    X = poly.fit_transform(df[['Day']]) # Transforms the 'Day' column of the dataframe to polynomial features
    return X # Returns the transformed features

def train_model(X, y, alpha, degree):
    """
    Train the Ridge regression model.
    """
    model = Ridge(alpha=alpha) # Initializes the Ridge regression model with the given alpha
    model.fit(X, y) # Fits the model on the data
    return model # Returns the trained model

def evaluate_model(model, X, y):
    """
    Evaluate the trained model using RMSE.
    """
    y_pred = model.predict(X) # Predicts the targets using the model and feature matrix X
    rmse = np.sqrt(mean_squared_error(y, y_pred)) # Calculates the RMSE between the true and predicted target vectors
    return rmse # Returns the RMSE

def cross_validation(X, y, degrees, alphas, cv_folds=5):
    """
    Perform cross-validation to tune hyperparameters.
    """
    best_rmse = float('inf') # Initializes the best RMSE to infinity for minimization
    # Initializes variables to store the best hyperparameters and model
    best_degree = None
    best_alpha = None
    best_model = None
    
    # Initializes KFold with the number of splits and without shuffling, since we want to keep the temporal ordering of the data
    kf = KFold(n_splits=cv_folds, shuffle=False) # 

    # Iterates over all combinations of degrees and alphas
    for degree in degrees:
        X_poly = generate_features(pd.DataFrame(X, columns=['Day']), degree) # Generates polynomial features for the current degree
        for alpha in alphas:
            model = Ridge(alpha=alpha) # Initializes a Ridge model for the current alpha
            avg_rmse = 0 # Initializes a variable to accumulate RMSE over all folds
            # Performs the KFold cross-validation
            for train_index, val_index in kf.split(X_poly):
                # Splits the data into training and validation sets
                X_train, X_val = X_poly[train_index], X_poly[val_index]
                y_train, y_val = y[train_index], y[val_index]
                model_clone = clone(model) # Clones the model to avoid contamination between folds
                model_clone.fit(X_train, y_train) # Fits the cloned model on the training set
                y_pred = model_clone.predict(X_val) # Predicts on the validation set
                avg_rmse += np.sqrt(mean_squared_error(y_val, y_pred)) / cv_folds # Accumulates the RMSE over all folds
            # Updates the best RMSE, hyperparameters, and model if the current model has a lower average RMSE
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_degree = degree
                best_alpha = alpha
                best_model = clone(model).fit(X_poly, y)  # Fits the best model on the entire dataset
    
    return best_model, best_degree, best_alpha, best_rmse # Returns the best model and its hyperparameters

def save_model(model, country, model_type, dataset):
    """
    Save the model to the ./models directory with a name reflecting the country and the dataset.
    """
    directory = '../models' # Defines the directory to save models to
    if not os.path.exists(directory): # Creates the directory if it does not exist
        os.makedirs(directory)
    filename = f'{directory}/{country}_{model_type}_{dataset}.joblib' # Defines the filename using the country, model type, and dataset
    joblib.dump(model, filename) # Saves the model to the specified filename using joblib format