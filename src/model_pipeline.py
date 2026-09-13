import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import os

def train_pipeline():
    data_path = 'data/tcga_expression_matrix.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError("Run data_loader.py first.")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Target_Clinical_Risk', 'Patient_ID'])
    y = df['Target_Clinical_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Pipeline trained. Test Accuracy: {acc:.2f} | ROC-AUC: {auc:.2f}")
    
    os.makedirs('model', exist_ok=True)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
    with open('model/features.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

if __name__ == "__main__":
    train_pipeline()
