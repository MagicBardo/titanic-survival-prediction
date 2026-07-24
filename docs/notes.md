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

* [x] Load the dataset
* [x] Understand every column
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

* [x] Read the Kaggle data description
* [x] Run `df.head()`
* [x] Run `df.info()`
* [x] Run `df.describe()`

### Useful Code

```python
import pandas as pd

df = pd.read_csv("train.csv")

df.head()
df.info()
df.describe()
```

### Column Understanding Table

| Column      | What it means     | Numeric / Categorical | Keep? |
| ----------- |-------------------|-----------------------|-------|
| PassengerId | numeration        | numeric               | no    |
| Survived    | alive or not      | categorical           | yes   |
| Pclass      | proxy for SES     | categorical           | yes   |
| Name        | name              | categorical           | no    |
| Sex         | sex               | categorical           | yes   |
| Age         | age               | numerical             | yes   |
| SibSp       | # of sibs/spouses | numerical             | yes   |
| Parch       | # of parents/kids | numerical             | yes   |
| Ticket      | ticket number     | numerical             | no    |
| Fare        | passenger fare    | numerical             | no    |
| Cabin       | cabin number      | numerical             | no    |
| Embarked    | entry port        | categorical           | yes   |

### My Notes

* Which columns are immediately understandable?
  * survival, pclass, sex, age, embarked
* Which columns confuse me?
  * sibsp, parch, cabin
* Which columns seem most likely to influence survival?
  * sex, age, pclass?

---

# Phase 2 — Explore the Data

## Goal

Find problems in the dataset.

### Tasks

* [x] Check missing values
* [x] Check duplicates
* [x] Inspect distributions
* [x] Look for impossible values

### Useful Code

```python
df.isna().sum()
df.duplicated().sum()

df["Age"].hist()
df["Fare"].hist()
```

### Missing Values

| Column   | Missing Count | % Missing | My Decision                                     |
| -------- | ------------: |----------:|-------------------------------------------------|
| Age      |           177 |   19.87 % | fill with median (because left tweek)           |
| Cabin    |           687 |   77.10 % | drop and add col "has cabin"                    |
| Embarked |             2 |    0.22 % | assign both C because high fare and same cabin  |

### My Notes

* Which column has the most missing values?
  * cabin
* Does missing data have a meaning? (e.g. no cabin assigned)
  * person was illegally on board, cabin wasn't recorded  
* What surprised me?
  * so many passangers have no cabin assigned, mostly in 3rd class

---

# Phase 3 — Clean the Data

## Goal

Create a consistent dataset.

### Tasks

* [x] Remove duplicates
* [x] Fill missing values
* [x] Drop useless columns
* [x] Fix data types

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

### My Notes

* Why did I choose median instead of mean?
  * because of the left tweek → not a normal distribution
* Why did I drop PassengerId?
  * because there is the index of the dataframe (basically the same)
* Am I unsure about any column?
  * no

---

# Phase 4 — Feature Engineering

## Goal

Create better information for the model.

### Tasks

* [x] Create FamilySize
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

| New Feature | Formula / Source             | Why it may help            |
|-------------|------------------------------|----------------------------|
| Family_Size | SibSp + Parch + 1            | combines cols              |
| Title       | Extracted from Name          | shows age and social group |
| Age_Group   | different borders (see code) | more precises grouping     |
| Has_Cabin   | cabin number -> 1, NaN -> 0  | fixed NaN in cabin         |

### My Notes

* Did the new feature make intuitive sense?
  * mostly
* Would a human use this information?
  * probably in a bit different formatting, but yes

---

# Phase 5 — Encode Categorical Features

## Goal

Convert text into numbers.

### Tasks

* [x] Identify text columns
* [x] Encode them
* [x] Verify all remaining columns are numeric

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

* Title

### My Notes

* Why can't ML models use raw text?
  * Don't know what text is, just knows numbers and their correlation
* What does one-hot encoding actually create?
  * splits column into multiple columns with all the options and just ones and zeros or True and False

---

# Phase 6 — Separate Features and Target

## Goal

Split questions from answers.

### Tasks

* [x] Create X
* [x] Create y
* [x] Check shapes

### Useful Code

```python
X = df.drop(columns=["Survived"])
y = df["Survived"]

print(X.shape)
print(y.shape)
```

### My Notes

**In my own words:**

* X = data that influences the survival 
* y = the parameter that is influenced

---

# Phase 7 — Train/Test Split

## Goal

Create a fair evaluation.

### Tasks

* [x] Import train_test_split
* [x] Split the data
* [x] Verify sizes

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
  * partial size of the test data relative to the whole dataset, rest is training data
* What does `random_state=42` do?
  * the lines for test data are chosen randomly, but the product needs to be reproducible (always in data science), so you need a relative constant
* Why must the test set stay untouched?
  * it is random and shouldn't be influenced and not seen by the program ever

---

# =========================

# MACHINE LEARNING STARTS

# =========================

# Phase 8 — Choose a Model

## Goal

Pick a first classifier.

### Tasks

* [x] Import DecisionTreeClassifier
* [x] Create the model object

### Useful Code

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
```

### My Notes

* Why is this a **classification** problem?
  * because we coose between *survived* and *not survived* (two Categories → categorical)
* What kind of output will the model produce?
  * binary (0 or 1)

---

# Phase 9 — Train the Model

## Goal

Let the model learn patterns.

### Tasks

* [x] Call `.fit()`
* [x] Confirm training completed

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

> The fit method creates rules for deciding if a passenger survives or not based on the given parameters

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
  * ids 709, 39
* Do I notice a pattern?
  * no, seems pretty random

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

Accuracy = 0.799 means that around 80% of the predictions were correct

This means: more than three quarters of the predictions are correct, which seems pretty good

### My Notes

* Is the accuracy better than random guessing?
  * yes
* Is one class predicted better than the other?
  * the 0 class has 89 correct predictions, class 1 just 54 → not that different but still a difference (0.82 vs 0.77)

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

| Experiment | What I changed                        |         Accuracy |
|------------|---------------------------------------|-----------------:|
| Baseline   | Default tree                          | 0.799 (89 vs 54) |
| Exp 1      | max_depth=4                           | 0.793 (92 vs 50) |
| Exp 2      | Added Family_Size                     | 0.777 (89 vs 50) |
| Exp 3      | Added Family_Size + <br/>max_depth=4  | 0.816 (89 vs 57) |
| Exp 4      | max_depth=10                          | 0.793 (88 vs 54) |
| Exp 5      | Added Family_Size + <br/>max_depth=10 | 0.788 (90 vs 51) |
| Exp 6      | max_depth=6                           | 0.782 (92 vs 48) |
| Exp 7      | Added Family_Size + <br/>max_depth=6  | 0.804 (94 vs 50) |
| Exp 8      | Added Family_Size + <br/>max_depth=5  | 0.816 (91 vs 55) |

### My Notes

* What change helped the most?
  * Adding the Family_Size together with the deeper tree, they were bad as a single change, but not too deep

---

# Phase 13 — Save the Model

### Useful Code

```python
import pickle

with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

### My Notes

* Why save the model instead of retraining every time?
  * because it needs time and is useless and needs space in working memory

---

# Final Reflection

## Explain these concepts without looking them up

* [x] Feature
* [x] Target
* [x] Training data
* [x] Test data
* [x] Classification
* [x] Generalization
* [x] Overfitting
* [x] Accuracy
* [x] One-hot encoding

## Biggest thing I learned
  
You can "overtrain" an ML (overfitting) when it thinks too deeply

---

## Biggest thing that confused me

Two changes on the model can both be bad when used alone but can improve the model when used together

---

## What I want to learn next

What the scikit-learn methods do in detail and how they actually work 

---
