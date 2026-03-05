import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def train_test_split_ts(df: pd.DataFrame, test_size=0.2):
    """
    Strict chronological split to prevent time-series data leakage.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def scale_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target_cols=None):
    """
    Scales the input features using StandardScaler fit on training data.
    """
    if target_cols is None:
        target_cols = ['failure_event', 'Leak Status', 'Burst Status', 'Bridge_Condition', 'Infrastructure_Defect']
        
    # filter targets that actually exist in the dataframe
    actual_targets = [c for c in target_cols if c in train_df.columns]
    
    scaler = StandardScaler()
    feature_cols = [c for c in train_df.columns if c not in actual_targets]
    
    train_feat = scaler.fit_transform(train_df[feature_cols])
    test_feat = scaler.transform(test_df[feature_cols])
    
    train_scaled = pd.DataFrame(train_feat, columns=feature_cols, index=train_df.index)
    test_scaled = pd.DataFrame(test_feat, columns=feature_cols, index=test_df.index)
    
    for c in actual_targets:
        train_scaled[c] = train_df[c].values
        test_scaled[c] = test_df[c].values
    
    return train_scaled, test_scaled, scaler, actual_targets

def create_sliding_windows(df: pd.DataFrame, window_size: int, target_cols: list):
    """
    Creates overlapping sliding windows for LSTM.
    Returns X of shape (num_samples, window_size, num_features)
    and y of shape (num_samples, num_targets) which are the target labels at the end of the window.
    """
    feature_cols = [c for c in df.columns if c not in target_cols]
    data_values = df[feature_cols].values
    target_values = df[target_cols].values
    
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(data_values[i:(i + window_size)])
        y.append(target_values[i + window_size - 1]) # labels of the last step in the window
        
    return np.array(X), np.array(y)

def apply_smote_ts(X: np.ndarray, y: np.ndarray):
    """
    Applies SMOTE to time-series windows by flattening, oversampling, and reshaping.
    Because y can be multi-label now, SMOTE only supports single label directly, therefore
    we combine the targets to create a composite multiclass index for oversampling.
    """
    n_samples, window_size, n_features = X.shape
    
    X_flat = X.reshape(n_samples, -1)
    
    # Create composite class identifier for multi-target
    # E.g. [0,0,0] -> "000", [1,0,1] -> "101"
    str_y = np.array([''.join(row.astype(str)) for row in y])
    
    # Check class distributions to prevent k_neighbors error
    class_counts = pd.Series(str_y).value_counts()
    
    # Only keep classes that have at least 6 samples (default k_neighbors + 1)
    # the rest we cannot SMOTE over, so we filter them out of the SMOTE pool
    valid_classes = class_counts[class_counts >= 6].index
    
    # Separate the data into "valid for SMOTE" and "too rare to SMOTE"
    valid_mask = np.isin(str_y, valid_classes)
    X_valid = X_flat[valid_mask]
    y_valid = str_y[valid_mask]
    
    X_rare = X_flat[~valid_mask]
    y_rare = str_y[~valid_mask]
    
    # Apply SMOTE only to the valid combinations
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled_flat, y_resampled_str = smote.fit_resample(X_valid, y_valid)
    
    # Re-append the ultra-rare instances without oversampling them
    if len(X_rare) > 0:
        X_resampled_flat = np.vstack([X_resampled_flat, X_rare])
        y_resampled_str = np.concatenate([y_resampled_str, y_rare])
    
    # Recreate the numeric multi-label array
    y_resampled = np.array([[int(char) for char in s] for s in y_resampled_str])
    
    # Reshape back to 3D
    X_resampled = X_resampled_flat.reshape(y_resampled.shape[0], window_size, n_features)
    
    return X_resampled, y_resampled

if __name__ == "__main__":
    from src.data.loader import load_and_preprocess_data
    from src.features.engineering import create_features
    
    file_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\smart_grid_stability_augmented.csv"
    water_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\water_leak_detection_1000_rows.csv"
    print("Loading data...")
    df = load_and_preprocess_data(file_path, water_path)
    
    print("Engineering features...")
    df = create_features(df)
    
    print("Train test split...")
    train_df, test_df = train_test_split_ts(df)
    
    print("Scaling data...")
    train_scaled, test_scaled, scaler, actual_targets = scale_data(train_df, test_df)
    
    print(f"Detected targets: {actual_targets}")
    print("Creating sliding windows...")
    window_size = 10
    X_train, y_train = create_sliding_windows(train_scaled, window_size=window_size, target_cols=actual_targets)
    X_test, y_test = create_sliding_windows(test_scaled, window_size=window_size, target_cols=actual_targets)
    
    print(f"X_train shape before SMOTE: {X_train.shape}, y_train shape: {y_train.shape}")
    
    print("Applying SMOTE-TS...")
    X_train_smote, y_train_smote = apply_smote_ts(X_train, y_train)
    
    print(f"X_train shape after SMOTE: {X_train_smote.shape}, y_train shape: {y_train_smote.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("Preprocessing complete!")
