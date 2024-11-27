
# COVID-19 Time Series Modeling and Forecasting

This repository contains the implementation of a machine learning assignment designed to model and forecast COVID-19 case trends using time-series data and polynomial regression. It focuses on data cleaning, feature engineering, hyperparameter tuning, and model evaluation to optimize predictions. The project is structured to provide a clear workflow for handling real-world, noisy datasets and applying machine learning techniques effectively. It uses Python as the primary language and scikit-learn for modeling and evaluation, emphasizing modularity, reproducibility, and well-documented code.

This analysis was performed as part of the 1st Assignment for the "Machine Learning in Computational Biology" graduate course of the MSc Data Science & Information Technologies Master's programme (Bioinformatics - Biomedical Data Science Specialization) of the Department of Informatics and Telecommunications department of the National and Kapodistrian University of Athens (NKUA), under the supervision of professor Elias Manolakos, in the academic year 2023-2024.

---

## Main Workflow

1. **Data Cleaning**:
    - Extract COVID-19 case data for selected countries.
    - Create time-series subsets for analysis (e.g., 700 and 800 days).
    - Visualize trends using notebooks.
2. **Feature Engineering**:
    - Generate polynomial features for modeling using PolynomialFeatures.
3. **Modeling**:
    - Train Ridge regression models with hyperparameter tuning.
    - Use cross-validation to identify optimal parameters (degree of polynomial and regularization strength).
    - Save model instances for reproducibility.
4. **Evaluation**:
    - Compare models using RMSE on training and testing datasets.
    - Establish baseline models and compare them with optimized models.
    - Plot results for interpretability.
5. **Prediction**:
    - Load saved models for prediction on unseen data.

---

## Results Overview

The project demonstrated the effectiveness of Ridge regression with polynomial features for modeling COVID-19 case trends. It highlighted the importance of proper data preprocessing and hyperparameter optimization in achieving low prediction errors. Key results include:
- RMSE comparisons between baseline and optimized models.
- Visualization of predicted vs. true case trends for multiple countries.

---

## Installation and Usage

### Cloning the Repository

```sh
git clone https://github.com/GiatrasKon/COVID19-TimeSeries-Forecasting.git
```

### Package Dependencies

Ensure you have the following packages installed:

- pandas
- numpy
- joblib
- sklearn
- matplotlib
- seaborn

Install dependencies using:

```sh
pip install pandas matplotlib seaborn numpy scikit-learn joblib
```

### Repository Structure

- `notebooks/`: Jupyter notebooks for data cleaning and analysis.
- `src/`: Source code, for model training and evaluation.
- `models/`: Saved model instances.
    - `final_models/`: Best-performing models.
- `data/`: Input and processed datasets.
- `documents/`: Assignment description and professor's feedback.

### Usage

1. Data Cleaning:
    - Open the Jupyter notebook `notebooks/data_cleaning.ipynb` to load the raw COVID-19 dataset (`data/time_series_covid19_confirmed_global.csv`), clean it, and create processed datasets (`final_data_700_days.csv` and `final_data_800_days.csv`) stored in the `data/` directory.
    - Visualize the data trends for each country.
2. Model Training and Hyperparameter Tuning:
    - Use the `src/functions.py` file, which contains:
        - `generate_features()`: Create polynomial features for modeling.
        - `train_model()`: Train Ridge regression models.
        - `cross_validation()`: Perform hyperparameter tuning with cross-validation.
        - `save_model()`: Save the trained models.
    - Run the notebook `notebooks/model_analysis.ipynb` to:
        - Load cleaned datasets.
        - Train models for each country using the 700- and 800-day datasets.
        - Save the trained models in the `models/` directory.
        - Plot true vs. predicted values and calculate RMSE.
3. Model Evaluation:
    - The `src/evaluation.py` script evaluates saved models by:
        - Loading a specific model from `models/` or `final_models/`.
        - Making predictions on unseen data (e.g., days 801–900).
        - Printing or saving the predictions for further analysis.
4. Model Comparison:
    - Use `model_analysis.ipynb` to:
        - Compare baseline and optimized models using bar plots.
        - Generate visualizations for performance metrics like RMSE.
5. Final Model Selection:
    - Select the best models for each country (trained on either the 700- or 800-day dataset) and save them in the `final_models/` directory for future use.

---