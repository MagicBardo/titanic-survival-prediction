# Titanic Survival Prediction 🚢

A machine learning project that predicts whether a Titanic passenger survived based on passenger information.

This project is my first complete machine learning workflow, covering the process from raw data exploration to a trained and evaluated classification model.

The goal is not only to create a working model but to understand the principles behind data preprocessing, supervised learning, model training, and evaluation.

---

# Project Goal

Build a machine learning model that predicts:

```
0 → Passenger did not survive

1 → Passenger survived
```

given information about a passenger.

This is a **binary classification problem**.

---

# Dataset

Dataset:

**Titanic - Machine Learning from Disaster**

Source:

Kaggle Titanic Competition

The main dataset used:

```
data/raw/train.csv
```

The dataset contains information about passengers, including:

- passenger class
- age
- gender
- ticket information
- family information
- fare
- cabin information
- embarkation port

Target column:

```
Survived
```

---

# Machine Learning Workflow

The project follows this workflow:

```
Data collection
        ↓
Exploratory data analysis
        ↓
Data cleaning
        ↓
Feature engineering
        ↓
Feature encoding
        ↓
Train/test split
        ↓
Model training
        ↓
Model evaluation
        ↓
Model improvement
        ↓
Model saving
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Project Structure

- `notebooks/` contains exploratory analysis and experiments.
- `src/` contains reusable machine learning code.
- `scripts/` contains executable commands for downloading data, training, predicting, and resetting the project.
- `docs/` contains project notes and decisions.
- `pyproject.toml` and `uv.lock` define the reproducible Python environment.

```
titanic-survival-prediction/
│
├── docs/
│   └── notes.md
│
├── notebooks/
│   └── titanic_survival.ipynb
│
├── scripts/
│   ├── download.py
│   ├── train.py
│   ├── predict.py
│   └── reset.py
│
├── src/
│   ├── model.py
│   └── preprocessing.py
│
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

# How to Run

Using
```
pip
```

or
using
```
uv 
```


## 1. Clone the repository

```bash
git clone MagicBardo/titanic-survival-prediction
cd titanic-survival-prediction
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

or 

```bash
uv sync
```

## 3. Download the dataset

Run:

```bash
python scripts/download.py
```

or 

```bash
uv run python scripts/download.py
```

## 4. Open the notebook

Start Jupyter:

```bash
jupyter notebook
```

or 

```bash
uv run jupyter notebook
```

Open:

```
notebooks/titanic_survival.ipynb
```

## 5. Reset data folder (if needed)

Empties ```data/raw``` and ```data/preprocessed```

```bash
python scripts/reset.py
```

or 

```bash
uv run python scripts/reset.py
```

---

# Model

The project uses supervised learning.

Initial models:

- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression

Models are evaluated and compared using classification metrics.

---

# Results

Results will be added after completing the experiments.

Example:

```
Model:
DecisionTreeClassifier

Accuracy:
XX%

F1 Score:
XX%
```

---

# Learning Goals

During this project I learned:

- how to explore a dataset
- how to handle missing values
- how to prepare data for machine learning
- how classification models work
- how to train and evaluate models
- how to improve machine learning results

Detailed notes and decisions can be found in:

```
docs/learning_notes.md
```

---

# Future Improvements

Possible improvements:

- better feature engineering
- hyperparameter tuning
- model comparison
- creating a prediction API
- deploying the model

