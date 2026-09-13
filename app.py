import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="TCGA Disease Predictor",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = 'model/disease_predictor.pkl'
    feat_path = 'model/feature_names.pkl'
    if os.path.exists(model_path) and os.path.exists(feat_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(feat_path, 'rb') as f:
            features = pickle.load(f)
        return model, features
    return None, None

model, feature_names = load_model()

st.title("🧬 TCGA Machine Learning Disease Predictor")
st.markdown("Predict patient disease states or clinical risk based on genomic expression profiles.")

if model is None:
    st.error("Model files not found! Please run training script first.")
else:
    st.sidebar.header("Input Controls")
    input_mode = st.sidebar.radio("Choose Input Mode", ["Manual Slider Input", "Upload CSV File"])
    
    if input_mode == "Manual Slider Input":
        st.subheader("Adjust Gene Expression Values")
        input_data = {}
        
        # Display sliders in columns
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            col_idx = i % 3
            with cols[col_idx]:
                input_data[feat] = st.slider(feat, float(-5.0), float(15.0), float(5.0))
                
        input_df = pd.DataFrame([input_data])
        
        if st.button("Run Prediction", type="primary"):
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            
            st.divider()
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error(f"⚠️ High Risk / Disease State Detected (Confidence: {proba[1]*100:.2f}%)")
            else:
                st.success(f"✅ Low Risk / Disease-Free State (Confidence: {proba[0]*100:.2f}%)")
                
            st.bar_chart(pd.DataFrame({"Probability": proba}, index=["Low Risk", "High Risk"]))

    else:
        st.subheader("Batch Prediction via CSV Upload")
        uploaded_file = st.file_uploader("Upload CSV containing gene expression columns", type=['csv'])
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", batch_df.head())
            
            if all(f in batch_df.columns for f in feature_names):
                if st.button("Run Batch Prediction"):
                    preds = model.predict(batch_df[feature_names])
                    probas = model.predict_proba(batch_df[feature_names])[:, 1]
                    
                    batch_df['Prediction'] = ["High Risk" if p == 1 else "Low Risk" for p in preds]
                    batch_df['High_Risk_Probability'] = probas
                    
                    st.write("Results:", batch_df)
                    st.download_button("Download Predictions CSV", batch_df.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")
            else:
                st.error("Uploaded CSV columns do not match expected TCGA feature names.")
