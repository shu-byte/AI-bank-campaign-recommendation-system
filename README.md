🏦 AI Bank Campaign Recommendation System

An Intelligent Customer Segmentation & Recommendation Engine built for the Banking Sector.

📌 Overview

This system utilizes Unsupervised Machine Learning (K-Prototypes) to segment bank customers based on demographic and behavioral data. It then employs XGBoost & SHAP to provide explainable marketing strategies (Cross-Sell, Retention, Acquisition).

Key Features:

Hybrid Clustering: Handles both numerical and categorical data naturally.

MLOps Pipeline: Supports automated retraining and hot-swapping of models via the UI.

Advanced Visualization: Uses t-SNE manifold learning to visualize high-dimensional customer clusters in 2D.

Explainable AI (XAI): Provides real-time SHAP value analysis for every prediction.

🛠️ Tech Stack

Frontend: React.js, Recharts, Axios

Backend: FastAPI (Python), Uvicorn

Machine Learning: Scikit-Learn, KModes, XGBoost, SHAP

Data Processing: Pandas, NumPy

🚀 Quick Start

1. Prerequisites

Ensure you have Python 3.9+ and Node.js installed.

2. Setup Backend

cd server
pip install -r requirements.txt
# Initialize the model (train on default dataset)
python train_model.py
# Start the API
uvicorn main:app --reload


3. Setup Frontend

cd frontend
npm install
npm start


📂 Project Structure

CampaignV2/
├── server/                 # FastAPI Backend & ML Scripts
│   ├── main.py             # API Endpoints
│   ├── train_model.py      # ML Training Pipeline
│   ├── preprocessing.py    # Data Cleaning Logic
│   └── bank-additional-full.csv
└── frontend/               # React User Interface
    ├── src/
    │   └── App.js          # Main UI Logic
    └── ...


📊 Methodology

Data Ingestion: Loads mixed-type data (Age, Job, Euribor Rate).

Clustering: Applies K-Prototypes to find 3 distinct customer personas (Engaged, Savers, Prospects).

Manifold Learning: Projects the 20-dimensional feature space into 2D using t-SNE for visualization.

Inference: Classifies new users into these clusters and recommends the optimal marketing strategy.

Developed for Final Year Project 2025.
