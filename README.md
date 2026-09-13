# TCGA Machine Learning Disease Predictor

An end-to-end bioinformatics machine learning pipeline and interactive web application built with Python, Scikit-Learn, and Streamlit. It uses gene expression features (inspired by TCGA datasets) to classify patient clinical risk or disease states.

## Project Structure
- `prepare_data.py`: Generates/prepares the TCGA gene expression dataset.
- `train.py`: Trains a Random Forest classifier and saves model artifacts.
- `app.py`: Interactive Streamlit dashboard for real-time predictions and batch CSV scoring.
- `model/`: Saved model binaries.
- `data/`: Dataset storage.

## How to Run Locally

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data and train the model:
   ```bash
   python prepare_data.py
   python train.py
   ```

3. Launch the Streamlit web dashboard:
   ```bash
   streamlit run app.py
   ```
