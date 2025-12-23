import pandas as pd
import numpy as np
import pickle
from kmodes.kprototypes import KPrototypes
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
from preprocessing import load_and_clean_data

# --- CHANGE: Default to your new 'partial' dataset ---
def train_system(data_path='bank-additional-train.csv'):
    print(f"1. Loading and Cleaning Data from {data_path}...")
    try:
        df = load_and_clean_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback explanation if file is missing
        print(f"   Make sure '{data_path}' exists in the folder.")
        raise e
    
    X = df.drop(columns=['target'])
    
    # Identify numerical and categorical columns
    num_columns = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    cat_columns = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    
    model_columns = X.columns.tolist()
    cat_indices = [X.columns.get_loc(col) for col in cat_columns]

    print("2. Scaling Numerical Data...")
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[num_columns] = scaler.fit_transform(X[num_columns])
    
    print(f"3. Training K-Prototypes on {len(df)} rows...")
    # n_init=5 ensures a robust search
    kproto = KPrototypes(n_clusters=3, init='Cao', verbose=1, n_init=5)
    clusters = kproto.fit_predict(X_scaled.values, categorical=cat_indices)
    
    # --- CLUSTER ALIGNMENT LOGIC ---
    print("4. Aligning Cluster IDs to Standard Profiles...")
    
    raw_centroids = kproto.cluster_centroids_
    
    # Handle different centroid structures from kmodes versions
    if len(raw_centroids) == 2 and hasattr(raw_centroids[0], 'shape') and raw_centroids[0].shape[0] == 3:
        num_centroids = raw_centroids[0]
        centroids_for_metric = num_centroids
    else:
        centroids_for_metric = np.array(raw_centroids)

    idx_prev_ptr = num_columns.index('previous')
    idx_eur_ptr = num_columns.index('euribor3m')

    vec_previous = centroids_for_metric[:, idx_prev_ptr].astype(float)
    vec_euribor = centroids_for_metric[:, idx_eur_ptr].astype(float)
    
    # Rule 1: "The Engaged" (Target ID 0) has the highest 'previous' contact history
    engaged_id = int(np.argmax(vec_previous))
    
    # Rule 2: "The Savers" (Target ID 1) have the lowest Interest Rate (Euribor) of the remaining
    remaining_ids = [i for i in range(3) if i != engaged_id]
    euribor_values = vec_euribor[remaining_ids]
    saver_id = remaining_ids[np.argmin(euribor_values)]
    
    # Rule 3: "New Prospects" (Target ID 2) is the last one
    prospect_id = [i for i in remaining_ids if i != saver_id][0]
    
    cluster_map = {
        engaged_id: 0,   # Engaged -> 0
        saver_id: 1,     # Savers -> 1
        prospect_id: 2   # Prospects -> 2
    }
    
    print(f"   Mapping Found: {cluster_map}")

    print("5. Training XGBoost Explainer & Preparing Visualization...")
    X_encoded = pd.get_dummies(X)
    xgb_columns = X_encoded.columns.tolist()
    
    explainer_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    explainer_model.fit(X_encoded, clusters)
    
    # --- GENERATE VISUALIZATION DATA (t-SNE + KNN) ---
    print("   Generating t-SNE 2D Cluster Map (this may take a moment)...")
    
    df_viz = X_encoded.copy()
    df_viz[num_columns] = scaler.transform(X[num_columns])
    df_viz['cluster_raw'] = clusters
    
    # Sample balanced set for visualization (Limit to ~1500 for speed)
    sample_indices = df_viz.groupby('cluster_raw', group_keys=False).apply(
        lambda x: x.sample(n=500, replace=True if len(x)<500 else False)
    ).index
    
    X_sample_for_viz = df_viz.loc[sample_indices].drop(columns=['cluster_raw'])
    y_cluster_sample = df_viz.loc[sample_indices, 'cluster_raw']
    
    # 1. Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(X_sample_for_viz)
    
    # 2. Train KNN Regressor
    viz_model = KNeighborsRegressor(n_neighbors=10)
    viz_model.fit(X_sample_for_viz, coords)
    
    # 3. Prepare viz data
    viz_df = pd.DataFrame(coords, columns=['x', 'y'])
    viz_df['cluster'] = y_cluster_sample.values
    viz_df['cluster'] = viz_df['cluster'].map(cluster_map)
    
    viz_points = viz_df.to_dict(orient='records')
    
    print("6. Saving Artifacts...")
    with open('cluster_model.pkl', 'wb') as f:
        pickle.dump(kproto, f)
    with open('explainer_model.pkl', 'wb') as f:
        pickle.dump(explainer_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('viz_model.pkl', 'wb') as f:
        pickle.dump(viz_model, f)
        
    metadata = {
        'columns': model_columns,
        'cat_indices': cat_indices,
        'xgb_columns': xgb_columns,
        'num_columns': num_columns,
        'cluster_map': cluster_map,
        'viz_points': viz_points 
    }
    with open('model_meta.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print("Done! System Retrained & Aligned.")

if __name__ == "__main__":
    train_system()