import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def generate_synthetic_tcga_data():
    np.random.seed(42)
    n_samples = 500
    n_features = 30
    
    # Generate random gene expression features
    feature_names = [f"GENE_{i+1}" for i in range(n_features)]
    X = np.random.randn(n_samples, n_features) * 2 + 5
    
    # Generate binary target: 0 (Low Risk/Disease Free), 1 (High Risk/Disease Progression)
    # Make target somewhat dependent on features
    logits = X[:, 0] * 0.8 - X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.5
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Target_Disease_State'] = y
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/tcga_mock_expression.csv', index=False)
    print("Dataset created successfully.")

if __name__ == "__main__":
    generate_synthetic_tcga_data()
