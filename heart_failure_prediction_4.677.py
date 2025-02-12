import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset not found: {filename}. Please make sure the dataset is in the correct directory.")
    df = pd.read_csv(filename)
    return df

# Compute correlation heatmaps
def plot_correlation_heatmaps(df):
    df_survivors = df[df['DEATH_EVENT'] == 0]
    df_deceased = df[df['DEATH_EVENT'] == 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(df_survivors.corr(), cmap='coolwarm', annot=True, ax=axes[0])
    axes[0].set_title('Correlation Heatmap (Survivors)')
    
    sns.heatmap(df_deceased.corr(), cmap='coolwarm', annot=True, ax=axes[1])
    axes[1].set_title('Correlation Heatmap (Deceased)')
    
    plt.tight_layout()
    plt.show()

# Split dataset into training and test sets
def split_data(df):
    return train_test_split(df['platelets'], df['serum_sodium'], test_size=0.5, random_state=123)

# Train polynomial regression models
def fit_polynomial_regression(x_train, y_train, x_test, y_test, degree):
    coeffs = np.polyfit(x_train, y_train, degree)
    model = np.poly1d(coeffs)
    y_pred = model(x_test)
    sse = np.sum((y_test - y_pred) ** 2)
    
    plt.scatter(x_test, y_test, color='blue', label='Actual')
    plt.scatter(x_test, y_pred, color='red', label='Predicted')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.xlabel('Platelets')
    plt.ylabel('Serum Sodium')
    plt.legend()
    plt.show()
    
    return sse

# Main script execution
if __name__ == "__main__":
    file_path = 'heart_failure_clinical_records_dataset.csv'  # Ensure this file is present
    df = load_data(file_path)
    
    # Select relevant features
    df = df[['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets', 'DEATH_EVENT']]
    
    # Generate correlation heatmaps
    plot_correlation_heatmaps(df)
    
    # Split data
    X_train, X_test, Y_train, Y_test = split_data(df)
    
    # Evaluate regression models
    degrees = [1, 2, 3]  # Linear, Quadratic, Cubic
    sse_results = {}
    
    for degree in degrees:
        sse = fit_polynomial_regression(X_train, Y_train, X_test, Y_test, degree)
        sse_results[f'Degree {degree}'] = sse
    
    print("SSE Results:", sse_results)
