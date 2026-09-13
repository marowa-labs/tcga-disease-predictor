import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle
import os

def train_model():
    data_path = 'data/tcga_mock_expression.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError("Run prepare_data.py first.")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Target_Disease_State'])
    y = df['Target_Disease_State']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained successfully! Test Accuracy: {acc:.2f}")
    
    os.makedirs('model', exist_ok=True)
    with open('model/disease_predictor.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/feature_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

if __name__ == "__main__":
    train_model()
