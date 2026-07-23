## 🚢 Titanic ML Project — Working File

> **Purpose:** This file is your project diary.
> Fill in the note sections while you work. Do not try to complete everything at once.

---

# Project Information

**Dataset:** Titanic — Machine Learning from Disaster

**Target column:** `Survived`

**Problem type:** Classification (predicting 0 or 1)

**Date started:** 23rd of Juli 2026

---

# Success Criteria

* [ ] Load the dataset
* [ ] Understand every column
* [ ] Clean missing values
* [ ] Encode categorical features
* [ ] Create at least one engineered feature
* [ ] Split the data correctly
* [ ] Train a classifier
* [ ] Evaluate accuracy
* [ ] Explain the results in your own words

---

# Phase 1 — Understand the Dataset

## Goal

Understand what every column means.

### Tasks

* [ ] Read the Kaggle data description
* [ ] Run `df.head()`
* [ ] Run `df.info()`
* [ ] Run `df.describe()`

### Useful Code

```python
import pandas as pd

df = pd.read_csv("train.csv")

df.head()
df.info()
df.describe()
```

### Column Understanding Table

| Column      | What it means | Numeric / Categorical | Keep? |
| ----------- | ------------- | --------------------- | ----- |
| PassengerId |               |                       |       |
| Survived    |               |                       |       |
| Pclass      |               |                       |       |
| Name        |               |                       |       |
| Sex         |               |                       |       |
| Age         |               |                       |       |
| SibSp       |               |                       |       |
| Parch       |               |                       |       |
| Ticket      |               |                       |       |
| Fare        |               |                       |       |
| Cabin       |               |                       |       |
| Embarked    |               |                       |       |

### My Notes

* Which columns are immediately understandable?
* Which columns confuse me?
* Which columns seem most likely to influence survival?

---

# Phase 2 — Explore the Data

## Goal

Find problems in the dataset.

### Tasks

* [ ] Check missing values
* [ ] Check duplicates
* [ ] Inspect distributions
* [ ] Look for impossible values

### Useful Code

```python
df.isna().sum()
df.duplicated().sum()

df["Age"].hist()
df["Fare"].hist()
```

### Missing Values

| Column   | Missing Count | % Missing | My Decision |
| -------- | ------------: | --------: | ----------- |
| Age      |               |           |             |
| Cabin    |               |           |             |
| Embarked |               |           |             |

### My Notes

* Which column has the most missing values?
* Does missing data have a meaning? (e.g. no cabin assigned)
* What surprised me?

---

# Phase 3 — Clean the Data

## Goal

Create a consistent dataset.

### Tasks

* [ ] Remove duplicates
* [ ] Drop useless columns
* [ ] Fill missing values
* [ ] Fix data types

### Useful Code

```python
# Remove duplicates
df = df.drop_duplicates()

# Fill Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop a column
df = df.drop(columns=["PassengerId"])
```

### Cleaning Decisions

| Column      | Action           | Why?              |
| ----------- | ---------------- | ----------------- |
| PassengerId | Drop             | Unique identifier |
| Age         | Fill with median |                   |
| Embarked    | Fill with mode   |                   |
| Cabin       | ?                |                   |

### My Notes

* Why did I choose median instead of mean?
* Why did I drop PassengerId?
* Am I unsure about any column?

---

# Phase 4 — Feature Engineering

## Goal

Create better information for the model.

### Tasks

* [ ] Create FamilySize
* [ ] Extract Title from Name
* [ ] Consider age groups

### Useful Code

```python
# Family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Extract title
df["Title"] = df["Name"].str.extract(r',\\s*([^\\.]+)\\.')

# Quick check
df[["Name", "Title"]].head()
```

### Features I Created

| New Feature | Formula / Source    | Why it may help |
| ----------- | ------------------- | --------------- |
| FamilySize  | SibSp + Parch + 1   |                 |
| Title       | Extracted from Name |                 |

### My Notes

* Did the new feature make intuitive sense?
* Would a human use this information?

---

# Phase 5 — Encode Categorical Features

## Goal

Convert text into numbers.

### Tasks

* [ ] Identify text columns
* [ ] Encode them
* [ ] Verify all remaining columns are numeric

### Useful Code

```python
# See text columns
df.select_dtypes(include="object").columns

# One-hot encode
df = pd.get_dummies(df, drop_first=True)

# Verify
df.dtypes
```

### Remaining Non-Numeric Columns

Write them here before encoding:

* ---
* ---
* ---

### My Notes

* Why can't ML models use raw text?
* What does one-hot encoding actually create?

---

# Phase 6 — Separate Features and Target

## Goal

Split questions from answers.

### Tasks

* [ ] Create X
* [ ] Create y
* [ ] Check shapes

### Useful Code

```python
X = df.drop(columns=["Survived"])
y = df["Survived"]

print(X.shape)
print(y.shape)
```

### My Notes

**In my own words:**

* X = ______________________________________
* y = ______________________________________

---

# Phase 7 — Train/Test Split

## Goal

Create a fair evaluation.

### Tasks

* [ ] Import train_test_split
* [ ] Split the data
* [ ] Verify sizes

### Useful Code

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Check

```python
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

### My Notes

* What does `test_size=0.2` mean?
* What does `random_state=42` do?
* Why must the test set stay untouched?

---

# =========================

# MACHINE LEARNING STARTS

# =========================

# Phase 8 — Choose a Model

## Goal

Pick a first classifier.

### Tasks

* [ ] Import DecisionTreeClassifier
* [ ] Create the model object

### Useful Code

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
```

### My Notes

* Why is this a **classification** problem?
* What kind of output will the model produce?

---

# Phase 9 — Train the Model

## Goal

Let the model learn patterns.

### Tasks

* [ ] Call `.fit()`
* [ ] Confirm training completed

### Useful Code

```python
model.fit(X_train, y_train)
```

### What happens during fit()?

* The model reads every passenger.
* It compares passenger features with the correct survival label.
* It builds decision rules.

### My Notes

Explain `.fit()` in one sentence:

> ---

---

# Phase 10 — Make Predictions

## Goal

Predict unseen passengers.

### Tasks

* [ ] Predict on X_test
* [ ] Compare with y_test

### Useful Code

```python
predictions = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Prediction": predictions
})

comparison.head(10)
```

### My Notes

* Which rows were wrong?
* Do I notice a pattern?

---

# Phase 11 — Evaluate the Model

## Goal

Measure performance.

### Tasks

* [ ] Calculate accuracy
* [ ] Calculate confusion matrix
* [ ] Interpret the results

### Useful Code

```python
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("Accuracy:", accuracy)
print(cm)
```

### Interpretation

Accuracy = ____________________

This means: ________________________________________

### My Notes

* Is the accuracy better than random guessing?
* Is one class predicted better than the other?

---

# Phase 12 — Improve the Model

## Ideas

* [ ] Tune tree depth
* [ ] Try Random Forest
* [ ] Improve feature engineering
* [ ] Revisit missing values

### Example

```python
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
```

### Experiments

| Experiment | What I changed   | Accuracy |
| ---------- | ---------------- | -------: |
| Baseline   | Default tree     |          |
| Exp 1      | max_depth=4      |          |
| Exp 2      | Added FamilySize |          |

### My Notes

What change helped the most?

---

---

# Phase 13 — Save the Model

### Useful Code

```python
import pickle

with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

### My Notes

Why save the model instead of retraining every time?

---

---

# Final Reflection

## Explain these concepts without looking them up

* [ ] Feature
* [ ] Target
* [ ] Training data
* [ ] Test data
* [ ] Classification
* [ ] Generalization
* [ ] Overfitting
* [ ] Accuracy
* [ ] One-hot encoding

## Biggest thing I learned

---

## Biggest thing that confused me

---

## What I want to learn next

---
