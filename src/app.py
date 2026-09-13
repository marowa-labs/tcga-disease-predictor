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
st.markdown("Advanced transcriptomic classification, PCA projection, and local feature contribution analysis.")

tabs = st.tabs(["Patient Risk Predictor", "Cohort PCA Explorer", "Global Feature Importance"])

with tabs[0]:
    st.subheader("Interactive Patient Risk Scoring & Explainability")
    
    # Preset Profiles
    profile_choice = st.radio("Select Baseline Clinical Profile:", ["Custom Control", "Healthy Control Profile", "Tumor Progressed Profile"], horizontal=True)
    
    col1, col2 = st.columns([1, 1])
    
    input_data = {}
    with col1:
        st.markdown("### Biomarker Expression Sliders (Top 10)")
        
        # Determine defaults based on profile
        if profile_choice == "Healthy Control Profile":
            defaults = df[df['Target_Clinical_Risk'] == 0][feature_names[:10]].mean()
        elif profile_choice == "Tumor Progressed Profile":
            defaults = df[df['Target_Clinical_Risk'] == 1][feature_names[:10]].mean()
        else:
            defaults = df[feature_names[:10]].median()
            
        for i, feat in enumerate(feature_names[:10]):
            val = float(defaults[feat])
            input_data[feat] = st.slider(feat, float(df[feat].min()), float(df[feat].max()), val)
            
        # Fill remaining features with median values
        for feat in feature_names[10:]:
            input_data[feat] = float(df[feat].median())
            
        input_df = pd.DataFrame([input_data])
        
    with col2:
        st.markdown("### Prediction & Local Explainability")
        
        pred = model.predict(input_df[feature_names])[0]
        proba = model.predict_proba(input_df[feature_names])[0]
        
        if pred == 1:
            st.error(f"⚠️ High Clinical Risk Detected (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ Low Risk / Stable Profile (Confidence: {proba[0]*100:.1f}%)")
            
        # Probability chart
        prob_df = pd.DataFrame({"Outcome": ["Low Risk", "High Risk"], "Probability": proba})
        fig = px.bar(prob_df, x="Outcome", y="Probability", color="Outcome", range_y=[0, 1], title="Classification Confidence")
        st.plotly_chart(fig, width="container")
        
        st.markdown("### Top Contributing Biomarkers (Local Impact)")
        # Calculate pseudo-local importance by multiplying input deviation from median with feature importances
        medians = df[feature_names[:10]].median()
        devs = [(feat, abs(input_data[feat] - medians[feat]) * imp) for feat, imp in zip(feature_names[:10], model.feature_importances_[:10])]
        devs.sort(key=lambda x: x[1], reverse=True)
        
        contrib_df = pd.DataFrame(devs[:5], columns=["Biomarker", "Impact Score"])
        fig_contrib = px.bar(contrib_df, x="Impact Score", y="Biomarker", orientation="h", title="Top Influential Genes for this Prediction")
        fig_contrib.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_contrib, width="container")

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
    st.plotly_chart(fig_pca, width="container")

with tabs[2]:
    st.subheader("Global Feature Importance (Random Forest)")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"Gene": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    
    fig_imp = px.bar(imp_df.head(15), x="Importance", y="Gene", orientation="h", title="Top 15 Predictive Biomarkers")
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, width="container")
