import os
import sys
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb
import shap

# Ensure local imports work by adding project root to sys path
sys.path.insert(0, r"C:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE")
from src.data.loader import load_and_preprocess_data
from src.features.engineering import create_features
from src.data.preprocessing import create_sliding_windows, train_test_split_ts, scale_data
from src.models.lstm_autoencoder import build_lstm_autoencoder, train_autoencoder, compute_reconstruction_error
from src.models.xgboost_risk_scorer import build_and_train_xgboost
from src.models.explainability import build_explainer, get_shap_values_for_instance

app = FastAPI(title="Micro-Failure Prediction API")

# Allow CORS for local HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
model_state = {
    "is_ready": False,
    "xgb_model": None,
    "explainer": None,
    "feature_names": None,
    "X_test_final": None,
    "y_test": None,
    "timestamps": None,
    "sim_index": 0
}

# Static mock assets
ASSETS = ["Grid Node G-01", "Bridge B-02", "Pump P-03"]

@app.on_event("startup")
def startup_event():
    """Loads and trains the model quickly in-memory for the demo API on startup."""
    print("Initializing ML Pipeline...")
    file_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\smart_grid_stability_augmented.csv"
    water_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\water_leak_detection_1000_rows.csv"
    bridge_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\bridge_data.csv"
    road_path = r"c:\Users\Ganesh\OneDrive\Desktop\AI-BASED MICRO-FAILURE PREDICTION IN URBAN INFRASTRUCTURE\infrastructure_defect_dataset.csv"
    
    # Check if we are running in the correct directory, else try relative or absolute
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. API might not function correctly.")
        return

    df = load_and_preprocess_data(file_path, water_path, bridge_path, road_path)
    df = create_features(df)
    
    # Use last 5000 rows for fast real-time API simulation
    df = df.iloc[-5000:]
    
    train_df, test_df = train_test_split_ts(df)
    train_scaled, test_scaled, scaler, actual_targets = scale_data(train_df, test_df)
    
    ws = 10
    X_train, y_train = create_sliding_windows(train_scaled, window_size=ws, target_cols=actual_targets)
    X_test, y_test = create_sliding_windows(test_scaled, window_size=ws, target_cols=actual_targets)
    
    n_features = X_train.shape[2]
    
    normal_indices = np.where(y_train.sum(axis=1) == 0)[0]
    X_train_normal = X_train[normal_indices]
    
    autoencoder = build_lstm_autoencoder(ws, n_features)
    train_autoencoder(autoencoder, X_train_normal, epochs=1, batch_size=128)
    
    train_recon_errors = compute_reconstruction_error(autoencoder, X_train)
    test_recon_errors = compute_reconstruction_error(autoencoder, X_test)
    
    X_train2d = X_train[:, -1, :] 
    X_train_final = np.column_stack([X_train2d, train_recon_errors])
    
    X_test2d = X_test[:, -1, :]
    X_test_final = np.column_stack([X_test2d, test_recon_errors])
    
    xgb_model = build_and_train_xgboost(X_train_final, y_train)
    explainer = build_explainer(xgb_model)

    feature_cols = [c for c in train_df.columns if c not in actual_targets]
    feature_names = feature_cols + ['lstm_anomaly_score']
    
    # Save to global state
    model_state["xgb_model"] = xgb_model
    model_state["explainer"] = explainer
    model_state["actual_targets"] = actual_targets
    model_state["feature_names"] = feature_names
    model_state["X_test_final"] = X_test_final
    model_state["y_test"] = y_test
    model_state["timestamps"] = test_df.index[ws:]
    model_state["is_ready"] = True
    print("Pipeline Ready! API is serving predictions.")


@app.get("/api/status")
def get_status():
    return {"status": "ready" if model_state["is_ready"] else "loading"}

@app.get("/api/simulation/advance")
def advance_simulation():
    if not model_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model initializing")
    
    if int(model_state["sim_index"]) < len(model_state["X_test_final"]) - 1:
        model_state["sim_index"] = int(model_state["sim_index"] + 1)
    
    return {"message": "advanced", "current_index": model_state["sim_index"]}

@app.get("/api/simulation/failure")
def simulate_failure():
    if not model_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model initializing")
    
    y_test = model_state["y_test"] # Shape corresponds to (simulated_timesteps, 5 targets)
    idx = int(model_state["sim_index"])
    
    # Find next time any of the 5 targets experienced an event
    future_failures = np.where(y_test[idx:].sum(axis=1) >= 1)[0]
    if len(future_failures) > 0:
        model_state["sim_index"] = int(model_state["sim_index"] + future_failures[0])
        return {"message": "jumped to failure", "current_index": model_state["sim_index"]}
        
    return {"message": "no further failures found", "current_index": int(idx)}

@app.get("/api/assets")
def get_assets():
    try:
        if not model_state["is_ready"]:
            raise HTTPException(status_code=503, detail="Model initializing")
            
        idx = int(model_state["sim_index"])
        X = model_state["X_test_final"]
        y = model_state["y_test"]
        t = model_state["timestamps"]
        xgb_model = model_state["xgb_model"]
        actual_targets = model_state["actual_targets"]
        
        current_instance = X[idx]
        actual_labels = y[idx]
        time_obs = t[idx].strftime("%Y-%m-%d %H:%M:%S")

        from src.models.xgboost_risk_scorer import predict_risk_score
        
        # Array of 5 probabilities 
        risk_arrays = predict_risk_score(xgb_model, current_instance.reshape(1, -1))[0]
        
        lstm_error = float(current_instance[-1])
        
        assets_data = []
        static_ids = ["Grid Node G-01", "Water Main W-04", "Pump P-03", "Bridge B-05", "Road Segment R-12"]
        
        # Extract the new objective features
        feature_names = model_state["feature_names"]
        maint_idx = feature_names.index('maintenance_history_days') if 'maintenance_history_days' in feature_names else -1
        usage_idx = feature_names.index('usage_frequency_score') if 'usage_frequency_score' in feature_names else -1
        
        maint_days = float(current_instance[maint_idx]) if maint_idx >= 0 else 0.0
        usage_score = float(current_instance[usage_idx]) if usage_idx >= 0 else 0.0
        
        for i, target_str in enumerate(actual_targets):
            current_risk = float(risk_arrays[i])
            is_anomaly = bool(actual_labels[i] == 1)
            trend = "Spiking" if current_risk > 60 else "Stable"
            
            # Predict early failure warning and maintenance recommendation based on risk
            early_failure_warning = bool(lstm_error > 0.05) or (current_risk > 75.0)
            
            if current_risk > 80:
                recommendation = "Immediate Inspection Required"
            elif current_risk > 50:
                recommendation = "Schedule Preventive Maintenance"
            else:
                recommendation = "Routine Monitoring"
            
            assets_data.append({
                "id": static_ids[i],
                "title": target_str,
                "risk_score": round(current_risk, 1),
                "trend": trend,
                "lstm_mse": round(lstm_error, 3),
                "last_updated": time_obs,
                "is_anomaly": is_anomaly,
                "target_idx": i,
                "maintenance_history_days": int(maint_days),
                "usage_frequency_score": round(usage_score, 1),
                "early_failure_warning": early_failure_warning,
                "maintenance_recommendation": recommendation
            })

        
        # Sort by risk score descending
        assets_data.sort(key=lambda x: x["risk_score"], reverse=True)
        
        raw_dict = {
            "timestamp": str(time_obs),
            "assets": assets_data,
            "ground_truth_failure": bool(any(actual_labels)),
            "sim_index": int(idx)
        }
        
        import json
        return json.loads(json.dumps(raw_dict, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    except Exception as e:
        import traceback
        return {"error": traceback.format_exc()}

@app.get("/api/asset/{asset_id}/shap")
def get_asset_shap(asset_id: str):
    if not model_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model initializing")
        
    # Map visual asset mock IDs back to their multi-class target indexes
    target_map = {
        "Grid Node G-01": 0,
        "Water Main W-04": 1,
        "Pump P-03": 2,
        "Bridge B-05": 3,
        "Road Segment R-12": 4
    }
    
    if asset_id not in target_map:
        return {
            "asset_id": asset_id,
            "features": ["N/A"],
            "values": [0]
        }
        
    target_idx = target_map[asset_id]
        
    idx = int(model_state["sim_index"])
    current_instance = model_state["X_test_final"][idx]
    explainer = model_state["explainer"]
    feature_names = model_state["feature_names"]
    
    # Calculate for the specific index output!
    shap_df = get_shap_values_for_instance(explainer, current_instance.reshape(1, -1), feature_names, target_idx)
    
    # Return top 6 features to display on the frontend chart
    top_shap = shap_df.head(6)
    
    return {
        "asset_id": asset_id,
        "features": top_shap['Feature'].tolist(),
        "values": top_shap['SHAP Value'].tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
