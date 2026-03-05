import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers time-series features such as rolling averages, volatility (std dev), 
    and momentum for the anomaly detection model across 5 combined structural domains.
    """
    df_feat = df.copy()
    
    # We will engineer features for the primary nodes to capture micro-degradations
    base_cols = ['tau1', 'p1', 'g1']
    
    # Adding Water Leak inputs
    if 'Pressure (bar)' in df_feat.columns:
        base_cols.extend(['Pressure (bar)', 'Flow Rate (L/s)'])
        
    # Adding Bridge continuous telemetry
    if 'Age_of_Bridge' in df_feat.columns:
        base_cols.extend(['Age_of_Bridge'])
        
    # Adding Road Defect continuous telemetry
    if 'Defect_Depth_mm' in df_feat.columns:
        base_cols.extend(['Defect_Length_mm', 'Defect_Width_mm', 'Defect_Depth_mm'])
    
    window_sizes = [3, 6] # Simulating 45min and 90min windows
    
    for col in base_cols:
        for w in window_sizes:
            # 1-6. Rolling Means and Std Devs
            df_feat[f'{col}_roll_mean_{w}'] = df_feat[col].rolling(window=w, min_periods=1).mean()
            df_feat[f'{col}_roll_std_{w}'] = df_feat[col].rolling(window=w, min_periods=1).std().fillna(0)
            
        # 7-9. Differences (first derivative)
        df_feat[f'{col}_diff'] = df_feat[col].diff().fillna(0)
        
        # 10-12. Momentum (ratio of current to rolling mean)
        # Add small epsilon to avoid division by zero
        df_feat[f'{col}_momentum'] = df_feat[col] / (df_feat[f'{col}_roll_mean_3'] + 1e-8)
        
        # 13-15. Acceleration (second derivative)
        df_feat[f'{col}_accel'] = df_feat[f'{col}_diff'].diff().fillna(0)

    # Drop any 'stab' column from features to prevent target leakage if not already dropped
    if 'stab' in df_feat.columns:
        df_feat.drop(columns=['stab'], inplace=True)
        
    # Drop rows with NaN if any slipped through (though min_periods=1 handles most)
    df_feat.dropna(inplace=True)
    
    return df_feat

if __name__ == "__main__":
    from src.data.loader import load_and_preprocess_data
    file_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\smart_grid_stability_augmented.csv"
    water_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\water_leak_detection_1000_rows.csv"
    bridge_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\bridge_data.csv"
    road_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\infrastructure_defect_dataset.csv"
    
    df = load_and_preprocess_data(file_path, water_path, bridge_path, road_path)
    df_features = create_features(df)
    print(f"Original shape: {df.shape}, New shape with features: {df_features.shape}")
    print(f"Engineered {df_features.shape[1] - df.shape[1]} new features.")
    print(df_features.head())
