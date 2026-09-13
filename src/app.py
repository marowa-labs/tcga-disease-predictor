import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

st.set_page_config(page_title="TCGA Multi-Omics Engine", page_icon="🧬", layout="wide")

@st.cache_resource
def load_assets():
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('model/features.pkl', 'rb') as f:
        features = pickle.load(f)
    df = pd.read_csv('data/tcga_expression_matrix.csv')
    return model, pca, features, df

model, pca, feature_names, df = load_assets()

st.title("🧬 TCGA Multi-Omics Clinical Risk & ML Engine")
st.markdown("Advanced transcriptomic classification, PCA projection, and feature contribution analysis.")

tabs = st.tabs(["Patient Risk Predictor", "Cohort PCA Explorer", "Feature Importance"])

with tabs[0]:
    st.subheader("Interactive Patient Risk Scoring")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Adjust Gene Expression Profiles")
        input_data = {}
        for feat in feature_names[:10]: # Showing top 10 sliders for clean UI
            input_data[feat] = st.slider(feat, float(df[feat].min()), float(df[feat].max()), float(df[feat].median()))
            
        # Fill remaining features with median values
        for feat in feature_names[10:]:
            input_data[feat] = float(df[feat].median())
            
        input_df = pd.DataFrame([input_data])
        
    with col2:
        st.markdown("### Prediction & Explainability")
        if st.button("Evaluate Patient Risk", type="primary"):
            pred = model.predict(input_df[feature_names])[0]
            proba = model.predict_proba(input_df[feature_names])[0]
            
            if pred == 1:
                st.error(f"⚠️ High Clinical Risk Detected (Confidence: {proba[1]*100:.1f}%)")
            else:
                st.success(f"✅ Low Risk / Stable Profile (Confidence: {proba[0]*100:.1f}%)")
                
            # Probability chart
            prob_df = pd.DataFrame({"Outcome": ["Low Risk", "High Risk"], "Probability": proba})
            fig = px.bar(prob_df, x="Outcome", y="Probability", color="Outcome", range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("TCGA Cohort Dimensionality Reduction (PCA)")
    X = df[feature_names]
    coords = pca.transform(X)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["Clinical Risk"] = df["Target_Clinical_Risk"].map({0: "Low Risk", 1: "High Risk"})
    pca_df["Patient ID"] = df["Patient_ID"]
    
    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color="Clinical Risk", hover_data=["Patient ID"],
        title="PCA Projection of TCGA Expression Matrix"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with tabs[2]:
    st.subheader("Global Feature Importance (Random Forest)")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"Gene": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    
    fig_imp = px.bar(imp_df.head(15), x="Importance", y="Gene", orientation="h", title="Top 15 Predictive Genes")
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)
