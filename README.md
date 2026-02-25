# Wine Quality Prediction using Multiple Linear Regression

## Overview

A supervised machine learning implementation that predicts wine quality scores using Multiple Linear Regression. This project demonstrates a complete ML workflow—from data preprocessing to model evaluation—using the UCI Wine Quality dataset for red wines.

The model estimates wine quality based on physicochemical attributes and identifies key factors influencing quality scores.

---

## Dataset

**Source:** [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

| Property | Value |
|----------|-------|
| **Records** | 1,599 |
| **Features** | 11 input variables |
| **Target** | `quality` (integer: 0-10) |

### Features Description

| Feature | Description |
|---------|-------------|
| `fixed_acidity` | Fixed acids content (g/dm³) |
| `volatile_acidity` | Volatile acids content (g/dm³) |
| `citric_acid` | Citric acid content (g/dm³) |
| `residual_sugar` | Residual sugar after fermentation (g/dm³) |
| `chlorides` | Salt content (g/dm³) |
| `free_sulfur_dioxide` | Free SO₂ content (mg/dm³) |
| `total_sulfur_dioxide` | Total SO₂ content (mg/dm³) |
| `density` | Wine density (g/cm³) |
| `pH` | Acidity level (0-14 scale) |
| `sulphates` | Sulphate content (g/dm³) |
| `alcohol` | Alcohol percentage (% vol) |

---

## Methodology

### 1. Data Preprocessing
- Load Wine Quality dataset
- Verify data types and structure
- Handle missing values (if present)
- Validate feature distributions

### 2. Feature Engineering
- All features are numeric (no encoding required)
- Separate target variable (`quality`) from feature set
- Optional: Feature scaling/normalization

### 3. Train-Test Split
```
Training Set: 80% (1,279 samples)
Testing Set:  20% (320 samples)
```
Ensures unbiased evaluation on unseen data.

### 4. Model Training

**Multiple Linear Regression:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- `y` = predicted quality score
- `β₀` = intercept
- `βᵢ` = learned coefficients
- `xᵢ` = feature values
- `ε` = residual error

### 5. Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **MSE** | `(1/n)Σ(yᵢ - ŷᵢ)²` | Average squared prediction error |
| **R²** | `1 - (SS_res / SS_tot)` | Proportion of variance explained |

---

## Project Structure
```
wine-quality-mlr/
│
├── data/
│   └── winequality-red.csv       # Dataset
│
├── notebooks/
│   └── exploratory.ipynb 
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading utilities
│   ├── preprocessing.py           # Data cleaning & splitting
│   ├── model.py                   # MLR training logic
│   ├── evaluate.py                # Performance metrics
│   └── visualize.py               # Plotting utilities
│
├── scripts/
│   └── download_data.ps1          # Automated data download
│
├── main.py                        # Main pipeline script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- pip package manager
- (Optional) PowerShell for Windows users

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/wine-quality-mlr.git
cd wine-quality-mlr
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

**Windows (PowerShell):**
```powershell
mkdir -Force .\data
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" -OutFile ".\data\winequality-red.csv"
```

**macOS/Linux:**
```bash
mkdir -p data
curl -o data/winequality-red.csv "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
```

### Step 5: Run Pipeline
```bash
python main.py
```

**Output:**
- Model performance metrics printed to console
- `residual_plot.png` saved to project root
- Trained model coefficients displayed

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Mean Squared Error (MSE)** | 0.3900 |
| **R² Score** | 0.4032 |

**Interpretation:** The R² score of 0.4032 indicates that approximately 40.32% of the variance in wine quality is explained by the physicochemical features. This moderate R² value suggests that while the linear model captures some relationships between chemical properties and quality, there are additional factors (potentially nonlinear relationships or unmeasured variables) that influence wine quality.

### Residual Analysis

A residual plot (`residual_plot.png`) is automatically generated to validate regression assumptions:
- **Random scatter:** Indicates linear relationship holds
- **Pattern/curve:** Suggests nonlinear relationships exist
- **Funnel shape:** Indicates heteroscedasticity

The residual plot helps diagnose:
- Linearity of the model
- Homoscedasticity (constant variance)
- Independence of errors

---

## Model Analysis

### Key Observations

1. **Model Performance:** The MSE of 0.39 indicates that, on average, predictions deviate from actual quality scores by approximately 0.62 points (√MSE).

2. **Explained Variance:** With 40.32% of variance explained, the model demonstrates moderate predictive capability. The remaining 59.68% of variance suggests:
   - Nonlinear relationships between features and quality
   - Potential feature interactions not captured
   - Influence of unmeasured factors (e.g., grape variety, wine age, storage conditions)

3. **Practical Implications:** The model can provide reasonable quality estimates for wine production monitoring, though additional features or more complex models may improve accuracy.

---

## Limitations

- **Linearity Assumption:** Model assumes linear relationships between features and target, which may not capture complex chemical interactions
- **Multicollinearity:** Correlated features (e.g., density and alcohol content) may inflate coefficient variance
- **Feature Interactions:** Does not capture nonlinear interactions without explicit engineering
- **Outlier Sensitivity:** Linear regression is sensitive to extreme values in both features and target
- **Limited Variance Explained:** R² of 0.40 suggests significant unexplained variance remains

---

## Future Enhancements

- [ ] Implement **Ridge** and **Lasso Regression** for regularization to handle multicollinearity
- [ ] Add **cross-validation** (k-fold) for robust performance estimation
- [ ] Engineer **polynomial features** to capture nonlinear relationships
- [ ] Perform **residual diagnostics** (Q-Q plots, Durbin-Watson test) for assumption validation
- [ ] Investigate **feature interactions** and **principal component analysis (PCA)**
- [ ] Compare with tree-based models (Random Forest, Gradient Boosting) for performance benchmarking
- [ ] Implement feature importance analysis to identify most influential predictors

---

## Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install via:
```bash
pip install -r requirements.txt
```

---

## Citation

**Dataset Source:**
```
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties.
Decision Support Systems, Elsevier, 47(4):547-553, 2009.
```

---

## Assignment Information

| Field | Details |
|-------|---------|
| **Student Name** | Odebiyi Aanuoluwapo |
| **Course Title** | [Cos-201] |
| **University** | [University Name] |
| **Academic Year** | 2026 |
| 
---

## Acknowledgments

- UCI Machine Learning Repository for providing the Wine Quality dataset
- scikit-learn contributors for the robust machine learning library
- Course instructors for project guidance and support

---

**Project Submitted:** 2026