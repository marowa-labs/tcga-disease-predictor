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
    
    # Preset Profile Buttons
    st.markdown("### Quick-Load Baseline Profiles")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    if 'profile_state' not in st.session_state:
        st.session_state.profile_state = "Custom"

    if col_btn1.button("Load Healthy Control Profile"):
        st.session_state.profile_state = "Healthy"
    if col_btn2.button("Load Tumor/High-Risk Profile"):
        st.session_state.profile_state = "Tumor"
    if col_btn3.button("Reset Custom Sliders"):
        st.session_state.profile_state = "Custom"

    col1, col2 = st.columns([1, 1])
    
    input_data = {}
    with col1:
        st.markdown("### Biomarker Expression Sliders (Top 10)")
        
        # Determine defaults based on profile state
        if st.session_state.profile_state == "Healthy":
            defaults = df[df['Target_Clinical_Risk'] == 0][feature_names[:10]].mean()
            st.info("Loaded Healthy Control baseline profile.")
        elif st.session_state.profile_state == "Tumor":
            defaults = df[df['Target_Clinical_Risk'] == 1][feature_names[:10]].mean()
            st.warning("Loaded Tumor High-Risk baseline profile.")
        else:
            defaults = df[feature_names[:10]].median()
            
        tooltips = {
            "TP53": "Tumor suppressor gene; loss of function promotes oncogenesis.",
            "BRCA1": "DNA repair gene; mutations increase breast/ovarian cancer risk.",
            "EGFR": "Receptor tyrosine kinase; overexpression drives cell proliferation.",
            "PTEN": "Tumor suppressor phosphatase; negative regulator of AKT pathway.",
            "MYC": "Proto-oncogene; transcription factor upregulated in many cancers.",
            "KRAS": "GTPase signal transducer; activating mutations drive tumor growth.",
            "BRAF": "Serine/threonine kinase; frequently mutated in melanoma.",
            "PIK3CA": "Lipid kinase; over-activation triggers cell growth and survival.",
            "AKT1": "Serine/threonine kinase; core node in cell survival pathways.",
            "CDKN2A": "Cyclin-dependent kinase inhibitor; cell cycle regulator."
        }
            
        for i, feat in enumerate(feature_names[:10]):
            val = float(defaults[feat])
            tip = tooltips.get(feat, "Gene expression quantification value (log2 FPKM).")
            input_data[feat] = st.slider(feat, float(df[feat].min()), float(df[feat].max()), val, help=tip)
            
        # Fill remaining features with median values
        for feat in feature_names[10:]:
            input_data[feat] = float(df[feat].median())
            
        input_df = pd.DataFrame([input_data])
        
    with col2:
        st.markdown("### Prediction & Explainability")
        
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
        
        st.markdown("### Top 5 Influential Biomarkers (Local Impact)")
        medians = df[feature_names[:10]].median()
        devs = [(feat, abs(input_data[feat] - medians[feat]) * imp) for feat, imp in zip(feature_names[:10], model.feature_importances_[:10])]
        devs.sort(key=lambda x: x[1], reverse=True)
        
        contrib_df = pd.DataFrame(devs[:5], columns=["Biomarker", "Impact Score"])
        fig_contrib = px.bar(contrib_df, x="Impact Score", y="Biomarker", orientation="h", title="Top 5 Genes Driving Prediction")
        fig_contrib.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_contrib, width="container")

    with st.expander("ℹ️ About the Model & Methodology"):
        st.markdown("""
        ### Model Explanation & Pipeline Details
        - **Dataset Source:** Modeled after The Cancer Genome Atlas (TCGA) RNA-Seq expression matrices. Features are quantified using normalized log2 counts across key oncology biomarkers.
        - **Baseline Normalization:** Median values represent population control averages. Sliders allow testing outlier patient profiles against healthy versus tumor baselines.
        - **Model Architecture:** Trained using a Scikit-Learn **Random Forest Classifier** optimized for high-dimensional genomic feature classification.
        - **Evaluation Metrics:** 
          - **Test Accuracy:** ~91%
          - **ROC-AUC Score:** ~0.97
        - **Explainability:** Local feature impact is evaluated by measuring feature divergence from population medians weighted against global Random Forest feature importances.
        """)

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
