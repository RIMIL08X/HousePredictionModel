# 🏠 HousePredictionModel

## 📌 Overview

This project implements and compares three regression models (**Linear Regression, Random Forest, and Gradient Boosting**) on the **Boston Housing dataset** to predict house prices. The project includes model training, evaluation, visualization, and model persistence.

![Code Execution](images/CodeExec.png)

---

## ⚙️ Prerequisites

* Python **3.11.9** (recommended via **Pyenv**)
* `pip` (Python package manager)

---

## 📂 Project Structure

```
HousePredictionModel/
│── images/                 # Visual outputs (plots, execution screenshot)
│   ├── CodeExec.png
│   ├── LinearRegression.png
│   ├── RandomForest.png
│   └── GradientBoosting.png
│── dumped_models/          # Saved trained models (.joblib files)
│── house_price.py          # Main code
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone this repository

```bash
git clone https://github.com/RIMIL08X/HousePredictionModel.git
cd HousePredictionModel
```

### 2️⃣ Set up Python 3.11.9 using Pyenv

```bash
# Install Pyenv (if not already installed)
curl https://pyenv.run | bash

# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Restart your shell or run:
exec "$SHELL"

# Install Python 3.11.9
pyenv install 3.11.9

# Set local Python version
pyenv local 3.11.9
```

### 3️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main script to train and evaluate all three models:

```bash
python house_price.py
```

The script will:

* Load and preprocess the **Boston Housing dataset**
* Train three regression models:

  * Linear Regression
  * Random Forest Regressor
  * Gradient Boosting Regressor (XGBoost)
* Evaluate each model using regression metrics
* Generate visualizations of the results
* Save trained models to `dumped_models/` as `.joblib` files
* Save performance plots to the `images/` directory

---

## 📊 Model Performance

### Linear Regression

![Linear Regression](images/LinearRegression.png)

### Random Forest

![Random Forest](images/RandomForest.png)

### Gradient Boosting

![Gradient Boosting](images/GradientBoosting.png)

---

## 📈 Results

The models are evaluated using:

* **Mean Squared Error (MSE)**
* **R-squared (R²) Score**

Visualizations include:

* Actual vs Predicted scatter plots

---

## 🔎 Model Comparison

* **Linear Regression** → Baseline model
* **Random Forest** → Robustness & feature importance
* **Gradient Boosting** → Potentially higher accuracy

---

## 🛠️ Customization

You can modify the model parameters in **`house_price.py`** to experiment with different configurations and observe their impact on prediction performance.

---

> 📝 **Made with ❤️ using Python 3.11.9**
