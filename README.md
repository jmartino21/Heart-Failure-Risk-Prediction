# Heart Failure Risk Prediction Using Regression Models

## Overview
This project explores the relationship between patient features and heart failure risk. It uses polynomial regression models to predict outcomes based on medical data, including platelet count, serum sodium, and creatinine phosphokinase levels.

## Features
- **Creatinine Phosphokinase (CPK)** – Enzyme levels
- **Serum Creatinine** – Indicator of kidney function
- **Serum Sodium** – Electrolyte balance measurement
- **Platelets** – Blood clotting levels
- **Death Event** – Patient survival status (0: Survived, 1: Deceased)

## Models Implemented
- Correlation heatmaps to explore feature relationships
- Polynomial regression models:
  - Linear Regression
  - Quadratic Regression
  - Cubic Spline Regression
  - Generalized Linear Model (GLM)

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset
This project requires the **heart_failure_clinical_records_dataset.csv** dataset. Ensure it is placed in the same directory as the script. If missing, you can download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records).

## Usage
### Running the Script
Execute the script using:
```bash
python heart_failure_prediction_4.677.py
```

### Steps Performed
1. Loads and preprocesses the dataset.
2. Generates correlation heatmaps to analyze feature relationships.
3. Splits the dataset into training and testing sets.
4. Trains polynomial regression models (linear, quadratic, cubic) and evaluates their performance.
5. Displays Sum of Squared Errors (SSE) for each model.
6. Plots prediction results for visualization.

## Output
- Correlation heatmaps comparing survivors and deceased patients.
- SSE scores for each regression model.
- Scatter plots of actual vs. predicted values for each model.

## License
This project is open-source and available for modification and use.

