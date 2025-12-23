import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    # Detect file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    print(f"   [Preprocessing] Loading {file_path} (Type: {ext})")
    
    try:
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            # Try semicolon first (typical for bank dataset), fallback to comma
            try:
                df = pd.read_csv(file_path, sep=';')
                # If only 1 column detected, it's probably the wrong separator
                if df.shape[1] <= 1:
                    print("   [Preprocessing] ';' separator yielded 1 column. Retrying with ','...")
                    df = pd.read_csv(file_path, sep=',')
            except:
                print("   [Preprocessing] Fallback to ',' separator...")
                df = pd.read_csv(file_path, sep=',')
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    df = df.drop_duplicates()
    
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
        
    if 'y' in df.columns:
        df['target'] = df['y'].map({'yes': 1, 'no': 0})
        df = df.drop(columns=['y'])
    
    # --- FIX: HANDLE PDAYS OUTLIER ---
    if 'pdays' in df.columns:
        # Create the binary flag first
        df['was_contacted'] = (df['pdays'] != 999).astype(int)
        
        # NOW change 999 to -1. 
        df['pdays'] = df['pdays'].replace(999, -1)
    
    # Standardize columns
    df.columns = [col.replace('.', '_') for col in df.columns]
    
    print(f"   [Preprocessing] Loaded shape: {df.shape}")
    return df