# Titanic Survival Prediction

A machine learning project that builds and evaluates models to predict passenger survival on the Titanic dataset. This project combines data exploration, feature engineering, and model training using scikit-learn and Jupyter notebooks.

---

## About the Project

This project analyzes the famous Titanic dataset and develops predictive models to determine which passengers were more likely to survive the disaster. The analysis includes:

- **Data Exploration & Visualization**: Understanding passenger demographics and survival patterns
- **Feature Engineering**: Creating meaningful features from raw data
- **Model Development**: Training and comparing multiple machine learning algorithms
- **Model Evaluation**: Assessing performance using appropriate metrics

This is more of a self-study project and for educational purposes. There might be a lot of errors and mistakes, but it is part of a studies journey. 

Feel free to leave hints and notes!

---

## Project Structure

```
titanic-survival-prediction/
├── README.md                          # This file
├── pyproject.toml                     # Project metadata and dependencies
├── requirements.txt                   # Pinned dependency versions
├── .python-version                    # Python version specification (3.12)
├── uv.lock                            # Locked dependencies for uv
│
├── notebooks/
│   └── titanic_survival.ipynb         # Analysis and testing notebook
│
├── data/                              # Dataset storage
│   ├── raw/                           # Original Titanic datasets
│   └── processed/                     # Cleaned and engineered features
│
├── models/                            # Trained model artifacts
│   └── *.pkl                          # Serialized scikit-learn models
│
├── src/
│   └── titanic_ml/                    # Python package for shared utilities
│
├── scripts/                           # Utility scripts
│
├── docs/                              # Documentation and task
│
└── logs/                              # Logs and output files
```

---

## Requirements

- Python 3.12+
- Dependencies listed in `pyproject.toml` / `requirements.txt`

### Key Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **kaggle**: Dataset downloading from Kaggle
- **jupyter/ipykernel**: Interactive notebooks

---

## Setup & Installation

### Option 1: Using `uv` (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. If you have it installed:

```bash
# Clone the repository
git clone https://github.com/MagicBardo/titanic-survival-prediction.git
cd titanic-survival-prediction

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start Jupyter
jupyter notebook
```

### Option 2: Using `pip`

```bash
# Clone the repository
git clone https://github.com/MagicBardo/titanic-survival-prediction.git
cd titanic-survival-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

---

## Usage

### Running the Analysis

1. Open `notebooks/titanic_survival.ipynb` in Jupyter:
   ```bash
   jupyter notebook notebooks/titanic_survival.ipynb
   ```

2. Execute the cells sequentially to:
   - Load and explore the Titanic dataset
   - Perform data preprocessing and feature engineering
   - Train multiple classification models
   - Evaluate model performance

    **Note**: This notebook it pretty messy and not built for readability. It was just meant for first exploration and testing before writing real scripts.

### Dataset

The Titanic dataset can be obtained from [Kaggle](https://www.kaggle.com/c/titanic). To download it manually:

```bash
# Set up Kaggle API credentials (see https://github.com/Kaggle/kaggle-api)
kaggle competitions download -c titanic
```

Or use the ```scripts/download.py``` file by running:

```bash
uv run python scripts/download.py
```

Raw data is placed in the `data/raw/` directory.

### Preprocessing & Training

When the data is downloaded, you can train the model with the automatically first processed data.

A direct analysis of the accuracy is logged in ```logs/```. The trained model will be saved in ```models/```.

To run all this, type:

```bash
uv run python scripts/train.py
```

### Predict

There is an example passenger hard coded in the ```scripts/predict.py``` file. The script predicts the life or death of this passenger. Play around with the values and look at the output. 

In the future there maybe will be a more user-friendly predicting usage. 

To see the result, run:

```bash
uv run python scripts/predict.py
```

### Reseting

To reset the project to the status of cloning but still with all packages installed, you can run

```bash
uv run python scripts/reset.py
```

With user acceptance, it deletes all files in ```data/raw```, ```data/processed```, ```models/``` and ```logs/```

---

## File Guide

| File/Directory | Purpose                                                      |
|---|--------------------------------------------------------------|
| `notebooks/titanic_survival.ipynb` | Messy Jupyter notebook for basic analysis and first training |
| `pyproject.toml` | Project metadata, dependencies, and build configuration      |
| `requirements.txt` | Exact dependency versions for reproducibility                |
| `models/` | Stores trained model files (`.joblib` format)                |
| `data/` | Raw and processed datasets                                   |
| `src/titanic_ml/` | Reusable Python modules and utilities                        |
| `scripts/` | Helper scripts for data processing or model evaluation       |
| `docs/` | Additional documentation                                     |
| `logs/` | Output logs and experiment tracking                          |

---

## Future Plans

- [ ] Add versioning of models
- [ ] Add cross-validation analysis
- [ ] Implement hyperparameter tuning
- [ ] Create model comparison visualizations
- [ ] Build an API for making predictions
- [ ] Add unit tests
- [ ] Deploy as a web service

---

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests!

---

## License

This project is open source and available under the MIT License.

___

**Questions or Issues?** Feel free to open an issue or contact the repository owner.