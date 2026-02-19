# Setup & Customization Guide

This guide is for anyone who wants to fork this repo and adapt it to a different dataset or regression problem with minimal changes.

---

## Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/medical-cost-mlr.git
cd medical-cost-mlr
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Swapping the Dataset

### Step 1 — Add your dataset
Place your `.csv` file in the `data/` folder:
```
data/your_dataset.csv
```

### Step 2 — Update `main.py`
Change the path passed to `load_data`:
```python
df = load_data("data/your_dataset.csv")
```

### Step 3 — Update `src/preprocessing.py`
This is the most important file to tweak. You need to:

**Change the target variable** (what you're predicting):
```python
# Original
x = df.drop("charges", axis=1)
y = df["charges"]

# Your version
x = df.drop("your_target_column", axis=1)
y = df["your_target_column"]
```

**Specify which columns are categorical** if `pd.get_dummies` doesn't pick them up automatically:
```python
# Original — encodes all object/bool columns automatically
df = pd.get_dummies(df, drop_first=True, dtype=int)

# If you want to control which columns get encoded
df = pd.get_dummies(df, columns=["col1", "col2", "col3"], drop_first=True, dtype=int)
```

**Adjust the train/test split ratio** if needed:
```python
# Original — 80% train, 20% test
return train_test_split(x, y, test_size=0.2, random_state=42)

# Example — 70% train, 30% test
return train_test_split(x, y, test_size=0.3, random_state=42)
```

> Changing `random_state` will give different splits but reproducible results. Set it to any integer.

---

## Running the Pipeline

Once your changes are in place:
```bash
python main.py
```

You will see:
```
Mean Squared Error: ...
R^2 Score: ...
```

A residual plot will also be saved as `residual_plot.png` in the project root.

---

## Printing Feature Coefficients

To see which features have the most influence on predictions, run:
```bash
python -c "
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model

df = load_data('data/your_dataset.csv')
x_train, x_test, y_train, y_test = preprocess_data(df)
model = train_model(x_train, y_train)
coefficients = pd.Series(model.coef_, index=x_train.columns)
print(coefficients.sort_values(ascending=False))
"
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Make sure your venv is activated and you ran `pip install -r requirements.txt` |
| `KeyError: 'your_target_column'` | Check the exact column name in your CSV — it's case sensitive |
| `ValueError` during get_dummies | Your dataset may have columns with mixed types — inspect with `df.dtypes` |
| Residual plot not saving | Make sure `matplotlib` is installed and the project root is writable |

---

## Requirements

All dependencies are listed in `requirements.txt`. To regenerate it after adding new packages:
```bash
pip freeze > requirements.txt
```

---

## Project Structure Reference

```
medical-cost-mlr/
│
├── data/                   # Place your dataset here
├── src/
│   ├── data_loader.py      # Change dataset path here
│   ├── preprocessing.py    # Main file to tweak for new datasets
│   ├── model.py            # Linear regression — no changes needed
│   ├── evaluate.py         # MSE and R² — no changes needed
│   └── visualize.py        # Residual plot — no changes needed
│
├── main.py                 # Entry point — update dataset path here
└── requirements.txt
```

---

*For questions or issues, open a GitHub Issue on the repository.*