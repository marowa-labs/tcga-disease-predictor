import pandas as pd
import numpy as np
import os

def generate_tcga_dataset():
    np.random.seed(42)
    n_samples = 600
    n_features = 40
    
    # Map synthetic genes to real oncology biomarkers
    biomarkers = [
        "TP53", "BRCA1", "EGFR", "PTEN", "MYC", 
        "KRAS", "BRAF", "PIK3CA", "AKT1", "CDKN2A",
        "ERBB2", "MTOR", "ATM", "RB1", "APC",
        "VHL", "JAK2", "STAT3", "NF1", "MET"
    ]
    other_genes = [f"GENE_{i+1:02d}" for i in range(n_features - len(biomarkers))]
    feature_names = biomarkers + other_genes
    
    # Simulate realistic gene expression with some clustered distributions
    X = np.random.normal(loc=6.0, scale=2.0, size=(n_samples, n_features))
    
    # Create non-linear interactions for clinical outcome
    risk_score = (
        X[:, 0] * 1.2 - 
        X[:, 1] * 0.9 + 
        X[:, 5] * 0.7 + 
        X[:, 12] * 1.5 - 
        np.mean(X[:, 20:], axis=1) * 0.5 +
        np.random.normal(0, 1, n_samples)
    )
    
    probs = 1 / (1 + np.exp(-risk_score))
    y = (probs > np.median(probs)).astype(int)
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Target_Clinical_Risk'] = y
    df['Patient_ID'] = [f"TCGA-PATIENT-{i+1000}" for i in range(n_samples)]
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/tcga_expression_matrix.csv', index=False)
    print("Enhanced multi-omics dataset generated successfully.")

if __name__ == "__main__":
    generate_tcga_dataset()
