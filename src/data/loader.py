import pandas as pd
import numpy as np

def load_and_preprocess_data(
    grid_file_path: str, 
    water_file_path: str = None,
    bridge_file_path: str = None,
    road_file_path: str = None
) -> pd.DataFrame:
    """
    Loads raw CSVs from all infrastructure domains and performs initial time-series alignment.
    """
    # 1. Load Grid Data (Base Timeline Anchor)
    df_grid = pd.read_csv(grid_file_path)
    
    # Simulate a time series index (assuming 15 min intervals)
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(minutes=15 * i) for i in range(len(df_grid))]
    df_grid['timestamp'] = timestamps
    df_grid.set_index('timestamp', inplace=True)
    
    # Convert 'stabf' categorically into numerical binary
    if 'stabf' in df_grid.columns:
        df_grid['failure_event'] = (df_grid['stabf'] == 'unstable').astype(int)
        df_grid.drop(columns=['stabf'], inplace=True)
        
    df_final = df_grid

    # 2. Load Water Data
    if water_file_path:
        df_water_raw = pd.read_csv(water_file_path)
        if 'Timestamp' in df_water_raw.columns:
            df_water_raw['Timestamp'] = pd.to_datetime(df_water_raw['Timestamp'])
            df_water_raw.set_index('Timestamp', inplace=True)
            
            num_cols = df_water_raw.select_dtypes(include=[np.number]).columns
            df_water_resampled = df_water_raw[num_cols].resample('15min').mean()
            
            target_cols = ['Leak Status', 'Burst Status']
            for col in target_cols:
                if col in df_water_raw.columns:
                    df_water_resampled[col] = df_water_raw[col].resample('15min').max().fillna(0).astype(int)
                    
            df_water_aligned = df_water_resampled.reindex(df_grid.index, method='ffill').fillna(0)
            df_final = pd.concat([df_final, df_water_aligned], axis=1)

    # 3. Load Bridge Data
    if bridge_file_path:
        df_bridge_raw = pd.read_csv(bridge_file_path)
        
        # Categorical Encoding
        if 'Material_Type' in df_bridge_raw.columns:
            df_bridge_raw['Material_Type'], _ = pd.factorize(df_bridge_raw['Material_Type'])
        if 'Maintenance_Level' in df_bridge_raw.columns:
            df_bridge_raw['Maintenance_Level'], _ = pd.factorize(df_bridge_raw['Maintenance_Level'])
            
        # Bridge data lacks timestamps. We artificially align it stretch-to-fit to our Grid Index.
        # We repeat the rows to match the length of our primary grid timeline simulation
        repeats = int(np.ceil(len(df_grid) / len(df_bridge_raw)))
        df_bridge_stretched = pd.concat([df_bridge_raw]*repeats, ignore_index=True).iloc[:len(df_grid)]
        df_bridge_stretched.index = df_grid.index # Lock to 15-min index
        
        df_final = pd.concat([df_final, df_bridge_stretched], axis=1)

    # 4. Load Road Defect Data
    if road_file_path:
        df_road_raw = pd.read_csv(road_file_path)
        
        # Standardize target names dynamically FIRST so our SMOTE targets logic catches them
        if 'Target_Defect_Class' in df_road_raw.columns:
            df_road_raw.rename(columns={'Target_Defect_Class': 'Infrastructure_Defect'}, inplace=True)
            
        # Categorical Encoding
        cats_to_encode = ['Infrastructure_Type', 'Defect_Location', 'Severity_Level', 'Lighting_Condition', 'Occlusion_Level', 'Inspection_Mode', 'Infrastructure_Defect']
        for col in cats_to_encode:
            if col in df_road_raw.columns:
                df_road_raw[col], _ = pd.factorize(df_road_raw[col])
                
        repeats = int(np.ceil(len(df_grid) / len(df_road_raw)))
        df_road_stretched = pd.concat([df_road_raw]*repeats, ignore_index=True).iloc[:len(df_grid)]
        df_road_stretched.index = df_grid.index # Lock to 15-min index
            
        df_final = pd.concat([df_final, df_road_stretched], axis=1)
            
    return df_final
