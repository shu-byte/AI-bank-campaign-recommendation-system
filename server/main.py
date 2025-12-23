from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import shap
import shutil
import os
from train_model import train_system # Import the training function

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models
models = {}

def load_models():
    """Helper to load/reload models from disk"""
    print("--- LOADING MODELS ---")
    try:
        with open('cluster_model.pkl', 'rb') as f:
            models['kproto'] = pickle.load(f)
        with open('explainer_model.pkl', 'rb') as f:
            models['xgb'] = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        with open('model_meta.pkl', 'rb') as f:
            models['meta'] = pickle.load(f)
        with open('viz_model.pkl', 'rb') as f:
            models['viz'] = pickle.load(f)
        print("MODELS LOADED SUCCESSFULLY")
    except FileNotFoundError:
        print("WARNING: Artifacts not found. System needs training.")

# Initial Load
load_models()

RECOMMENDATIONS = {
    0: "Strategy: Cross-Sell (Premium Package). High engagement potential.",
    1: "Strategy: Retention (Fixed Deposit). Conservative/Saver profile.",
    2: "Strategy: Acquisition (Intro Offer). Standard new customer."
}

class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

# --- FIX: Changed 'async def' to 'def' to run in threadpool ---
@app.post("/train_from_file")
def train_from_file(file: UploadFile = File(...)):
    """
    Uploads a CSV/Excel file, retrains the model, and reloads it.
    """
    temp_filename = f"temp_upload_{file.filename}"
    try:
        # 1. Save uploaded file temporarily
        with open(temp_filename, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved to {temp_filename}. Starting Training...")

        # 2. Trigger Training
        # Since this is now a synchronous 'def', FastAPI runs it in a separate thread
        train_system(data_path=temp_filename)

        # 3. Reload Models
        load_models()
        
        return {"status": "success", "message": "Model successfully retrained and reloaded!"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 4. Cleanup
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

@app.post("/predict_campaign")
def predict_campaign(data: CustomerData):
    if 'kproto' not in models:
        raise HTTPException(status_code=503, detail="Models not loaded. Please train the system first.")

    try:
        meta = models['meta']
        scaler = models['scaler']
        kproto_model = models['kproto']
        viz_model = models['viz']
        xgb_model = models['xgb']

        print("\n--- NEW PREDICTION REQUEST ---")
        input_dict = data.dict()
        
        # --- Smart Context Alignment ---
        euribor = input_dict.get('euribor3m', 4.857)
        if euribor > 3.0:
            input_dict['emp_var_rate'] = 1.1
            input_dict['cons_price_idx'] = 93.994
            input_dict['cons_conf_idx'] = -36.4
            input_dict['nr_employed'] = 5191.0
        else:
            input_dict['emp_var_rate'] = -1.8
            input_dict['cons_price_idx'] = 92.893
            input_dict['cons_conf_idx'] = -46.2
            input_dict['nr_employed'] = 5099.1

        if input_dict['poutcome'] == 'success' and input_dict['pdays'] == 999:
             input_dict['pdays'] = 6
        if input_dict['poutcome'] == 'nonexistent':
             input_dict['pdays'] = 999

        input_dict['was_contacted'] = 1 if input_dict['pdays'] != 999 else 0
        if input_dict['pdays'] == 999:
            input_dict['pdays'] = -1
            
        df_input = pd.DataFrame([input_dict])
        df_input = df_input[meta['columns']]
        
        # Scale for Prediction
        df_scaled = df_input.copy()
        num_cols = meta['num_columns']
        df_scaled[num_cols] = scaler.transform(df_input[num_cols])
        
        # Predict Cluster
        cat_indices = [int(x) for x in meta['cat_indices']]
        clusters = kproto_model.predict(df_scaled.values, categorical=cat_indices)
        original_cluster_id = int(clusters[0])
        
        # Apply Cluster Map
        cluster_map = meta.get('cluster_map', {0:0, 1:1, 2:2})
        final_cluster_id = cluster_map[original_cluster_id]
        
        # Create Aligned DataFrame for XGBoost & Visualization
        df_encoded = pd.get_dummies(df_input)
        df_aligned_xgb = pd.DataFrame(0.0, index=[0], columns=meta['xgb_columns'])
        common_cols = [c for c in df_encoded.columns if c in meta['xgb_columns']]
        df_aligned_xgb[common_cols] = df_encoded[common_cols]
        df_aligned_xgb = df_aligned_xgb.astype(float)
        
        # --- VISUALIZATION: PROJECT USER TO t-SNE MAP ---
        df_for_viz = df_aligned_xgb.copy()
        df_for_viz[num_cols] = scaler.transform(df_input[num_cols])
        
        user_coords = viz_model.predict(df_for_viz)
        user_x = float(user_coords[0][0])
        user_y = float(user_coords[0][1])

        # SHAP Analysis
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(df_aligned_xgb)
        
        if isinstance(shap_values, list):
            raw_vals = shap_values[original_cluster_id]
        else:
            raw_vals = shap_values
        
        raw_vals = raw_vals.flatten()
        abs_vals = np.abs(raw_vals)
        top_indices = np.argsort(abs_vals)[-5:][::-1]
        
        shap_data = []
        reasons = []
        
        for idx in top_indices:
            idx_int = int(idx)
            feat_name = meta['xgb_columns'][idx_int]
            feat_val = df_aligned_xgb.iloc[0, idx_int]
            impact_score = float(raw_vals[idx_int]) 
            clean_name = feat_name.replace('_', ' ').title()
            
            shap_data.append({
                "feature": clean_name,
                "value": feat_val,
                "impact": impact_score
            })
            reasons.append(f"{feat_name} = {feat_val}")
            
        return {
            "assigned_cluster": final_cluster_id,
            "recommendation": RECOMMENDATIONS.get(final_cluster_id, "Standard Service"),
            "key_factors": reasons,
            "shap_data": shap_data, 
            "explanation": f"Customer fits Standard Cluster {final_cluster_id}.",
            "visualization": {
                "user_point": {"x": user_x, "y": user_y, "cluster": final_cluster_id},
                "background_points": meta['viz_points']
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))