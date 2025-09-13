# boston_housing_regression.py
# First Install dependencies:
# Use pip install -r requirement's.txt

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

# Setup TensorBoard writer : This is to ensure all the metrics are graphically displayed
log_dir = os.path.join("runs", "boston_housing_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
writer = SummaryWriter(log_dir)

# Here the boston Housing Dataset is loaded 
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data
y = boston.target.astype(float)

print(f"Dataset shape: {X.shape}")

# Here we Scale the dataset Hence completing the preprocess
if X.isnull().sum().sum() > 0:
    X.fillna(X.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42
)

# Here we define each of the 3 models to be tested
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# dumping models in disk just to be sure
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Training, Predicting and Saving
results = {}

for step, (name, model) in enumerate(models.items(), start=1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    
# Logging metrics to TensorBoard
    writer.add_scalar(f"MSE/{name}", mse, step)
    writer.add_scalar(f"R2/{name}", r2, step)
    
# Logging predicted vs actual plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name} Predicted vs Actual")
    writer.add_figure(f"Predicted_vs_Actual/{name}", plt.gcf(), step)
    plt.close()
    
# Saving trained model to disk
    model_file = os.path.join(models_dir, f"{name.replace(' ', '_')}.joblib")
    joblib.dump(model, model_file)
    print(f"Saved {name} model to: {model_file}")

# Comparing Models
results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n", results_df)

# Ending the TensorBoard Writer
writer.close()
print(f"\nTensorBoard logs saved to: {log_dir}")
print("Run the following command to visualize:")
print(f"tensorboard --logdir={os.path.abspath('runs')}")
