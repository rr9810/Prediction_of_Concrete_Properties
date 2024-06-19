import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pyforest import *
lazy_imports()

data = pd.read_csv('reverse.csv')

# Prepare the data
X = data[['Compressive Strength (MPa)']]
y_wc = data['w/c']
y_fc = data['f/c']
y_cf = data['c/f']

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Split the data for water-cement ratio prediction
X_train_wc, X_test_wc, y_train_wc, y_test_wc = train_test_split(X, y_wc, test_size=0.2, random_state=42)
# Split the data for fine-coarse aggregate ratio prediction
X_train_fc, X_test_fc, y_train_fc, y_test_fc = train_test_split(X, y_fc, test_size=0.2, random_state=42)
# Split the data for cement-fine aggregate ratio prediction
X_train_cf, X_test_cf, y_train_cf, y_test_cf = train_test_split(X, y_cf, test_size=0.2, random_state=42)

# Initialize the Random Forest models
rf_wc = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fc = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the models
rf_wc.fit(X_train_wc, y_train_wc)
rf_fc.fit(X_train_fc, y_train_fc)
rf_cf.fit(X_train_cf, y_train_cf)

# Make predictions and evaluate
wc_predictions = rf_wc.predict(X_test_wc)
fc_predictions = rf_fc.predict(X_test_fc)
cf_predictions = rf_cf.predict(X_test_cf)

# Calculate and print MSE for evaluations
wc_mse = mean_squared_error(y_test_wc, wc_predictions)
fc_mse = mean_squared_error(y_test_fc, fc_predictions)
cf_mse = mean_squared_error(y_test_cf, cf_predictions)
wc_initial_rmse = np.sqrt(mean_squared_error(y_test_wc, wc_predictions))
fc_initial_rmse = np.sqrt(mean_squared_error(y_test_fc, fc_predictions))
cf_initial_rmse = np.sqrt(mean_squared_error(y_test_cf, cf_predictions))

# print(f'Water-Cement Ratio Prediction MSE: {wc_mse}')
# print(f'Fine to Coarse Aggregate Ratio Prediction MSE: {fc_mse}')
# print(f'Cement to Fine Aggregate Ratio Prediction MSE: {cf_mse}')
# print(f'RMSE for water/cement = {wc_initial_rmse}')
# print(f'RMSE for fine/coarse = {fc_initial_rmse}')
# print(f'RMSE for cement/fine = {cf_initial_rmse}')

# Function to predict ratios and calculate concrete components based on compressive strength
def predict_and_calculate_ratios(compressive_strength):
    input_data = pd.DataFrame({'Compressive Strength (MPa)': [compressive_strength]})
    predicted_wc = rf_wc.predict(input_data)[0]
    predicted_fc = rf_fc.predict(input_data)[0]
    predicted_cf = rf_cf.predict(input_data)[0]

    # Calculate the components based on the predicted ratios
    cement = 1  # Assume cement as 1 unit
    water = predicted_wc * cement  # water = w/c ratio * cement
    fine_aggregate = cement / predicted_cf  # fine_aggregate = cement / (c/f)
    coarse_aggregate = fine_aggregate / predicted_fc  # coarse_aggregate = fine_aggregate / (f/c)

    return (cement, water, fine_aggregate, coarse_aggregate)

# Manual input of compressive strength
# compressive_strength = float(input("Enter the compressive strength in MPa: "))
# water, cement, fine_aggregate, coarse_aggregate = predict_and_calculate_ratios(compressive_strength)

# print(f"Calculated Water : Cement : Fine Aggregate : Coarse Aggregate =   {water:.3f},  {cement:.3f},  {fine_aggregate:.3f},  {coarse_aggregate:.3f}")

#saving best model
# import pickle
# savePath = r'models\reverse.sav'
# pickle.dump(rf_wc,rf_fc,rf_cf,open(savePath, 'wb'))

# print(predict_and_calculate_ratios(10))
# print(predict_and_calculate_ratios(20))
# print(predict_and_calculate_ratios(30))
# print(predict_and_calculate_ratios(40))
# print(predict_and_calculate_ratios(50))
# print(predict_and_calculate_ratios(60))
# print(predict_and_calculate_ratios(70))
# print(predict_and_calculate_ratios(80))
# print(predict_and_calculate_ratios(90))
# print(predict_and_calculate_ratios(100))

