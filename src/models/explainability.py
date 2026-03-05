import shap
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier

def build_explainer(model: MultiOutputClassifier, background_data=None) -> list:
    """Builds a list of SHAP TreeExplainers, one for each target's underlying XGBoost model."""
    explainers = []
    # MultiOutputClassifier stores the individual models in the `estimators_` attribute
    for estimator in model.estimators_:
        # XGBoost TreeExplainer
        explainer = shap.TreeExplainer(estimator)
        explainers.append(explainer)
    return explainers

def get_shap_values_for_instance(explainers: list, instance: np.ndarray, feature_names: list, target_idx: int = 0) -> pd.DataFrame:
    """
    Calculates SHAP values for an instance for a specific target index.
    target_idx corresponds to the class (0 = Grid Failure, 1 = Leak, 2 = Burst)
    """
    explainer = explainers[target_idx]
    
    # SHAP explainer returns an Explanation object
    shap_vals = explainer(instance)
    
    # For binary classification XGBoost, shap_vals.values might be shape (1, num_features)
    values = shap_vals.values[0]
    
    df_shap = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': values,
        'Absolute SHAP': np.abs(values)
    })
    
    df_shap = df_shap.sort_values(by='Absolute SHAP', ascending=False)
    return df_shap
