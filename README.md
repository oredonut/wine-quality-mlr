Multiple Linear Regression on Medical Insurance Cost Dataset
Project Overview

This project implements a Supervised Machine Learning model to predict individual medical insurance costs using Multiple Linear Regression.

The objective is to demonstrate a complete machine learning workflow — from data preprocessing and categorical encoding to model training and evaluation — using a real-world dataset of 1,338 medical records.

The model estimates medical insurance charges based on demographic and health-related attributes and identifies key factors influencing healthcare expenditure.

Dataset Description

Total Records: 1,338

Total Features: 6 input variables + 1 target variable

Target Variable: charges — Individual medical insurance cost

Features
Feature	Description
Age	Age of the primary beneficiary
Sex	Gender of the insurance contractor (female, male)
BMI	Body Mass Index ($kg/m^2$)
Children	Number of dependents covered by insurance
Smoker	Smoking status (yes, no)
Region	Residential area in the US (northeast, southeast, southwest, northwest)
Methodology

This project follows a structured machine learning pipeline:

1. Data Preprocessing

Dataset loading and validation

Verification of data types

Handling of missing values (if present)

2. Feature Encoding

Categorical variables (sex, smoker, region) are converted into numerical format using One-Hot Encoding.

This prevents:

Implied ordinal relationships

Dummy variable trap (by dropping one category)

3. Data Partitioning

The dataset is split into:

80% Training Set

20% Testing Set

This ensures evaluation on unseen data to assess generalization performance.

4. Model Training

A Multiple Linear Regression model is trained using:

                y=β0​+β1​x1​+β2​x2​+...+βn​xn​+ϵ

Where:

𝛽
β represents learned coefficients

𝜖
ϵ represents residual error

5. Model Evaluation

The model is evaluated using:

Mean Squared Error (MSE)

R² Score (Coefficient of Determination)

Project Structure
medical-cost-mlr/
│
├── data/                 # Dataset files
├── notebooks/            # Exploratory Data Analysis (EDA)
├── src/                  # Modular source code
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
│
├── scripts/
│   └── download_data.sh  # Dataset download script
│
├── main.py               # Pipeline entry point
├── requirements.txt      # Dependencies
└── README.md

How to Run the Project
1. Clone the Repository
git clone https://github.com/smurftyy/medical-cost-mlr.git
cd medical-cost-mlr

2. Install Dependencies
pip install -r requirements.txt

3. Download the Dataset

Using curl:

mkdir -p data
curl -L -o data/insurance.csv \
https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv


Or run the script:

bash scripts/download_data.sh

4. Execute the Model
python main.py

Results

Mean Squared Error (MSE):
<INSERT_MSE_VALUE_HERE>

R² Score:
<INSERT_R2_VALUE_HERE>

Interpretation

The R² score indicates that approximately <INSERT_PERCENTAGE>% of the variance in medical insurance charges is explained by the model.

The feature with the strongest positive influence is: <INSERT_FEATURE_NAME>

Holding other variables constant, a one-unit increase in <INSERT_FEATURE> increases predicted charges by <INSERT_COEFFICIENT> units.

Limitations

Assumes a linear relationship between features and target.

Sensitive to multicollinearity.

Cannot capture nonlinear interactions without feature engineering.

Future Improvements

Implement Ridge and Lasso Regression for regularization.

Perform cross-validation for more robust evaluation.

Investigate feature interactions and polynomial regression.

Conduct residual diagnostics to validate regression assumptions.

Author

Name: <Your Name>
Course: <Course Title>
Institution: <Institution Name>
Year: <Year>