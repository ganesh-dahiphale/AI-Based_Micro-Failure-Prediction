import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def build_and_train_rf(X_train2d: np.ndarray, y_train: np.ndarray) -> MultiOutputClassifier:
    """
    Trains a MultiOutput Random Forest classifier for risk classification.
    """
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model = MultiOutputClassifier(base_model)
    model.fit(X_train2d, y_train)
    return model

def predict_rf_probability(model: MultiOutputClassifier, X_test2d: np.ndarray) -> np.ndarray:
    """
    Predicts risk classification probabilities.
    Returns: array of shape (n_samples, n_targets) containing probabilities scaled 0-100.
    """
    probs_list = model.predict_proba(X_test2d)
    
    # Extract the positive class probability for each target
    positive_probs = []
    for probs in probs_list:
        if probs.shape[1] > 1:
            positive_probs.append(probs[:, 1])
        else:
            # Handle edge case where a target only has 1 class in the batch
            positive_probs.append(np.zeros(probs.shape[0]))
            
    probs_array = np.column_stack(positive_probs)
    risk_scores = probs_array * 100
    return risk_scores
