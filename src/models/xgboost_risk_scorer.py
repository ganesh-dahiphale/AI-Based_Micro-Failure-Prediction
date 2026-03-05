import xgboost as xgb
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import average_precision_score, classification_report

def build_and_train_xgboost(X_train2d: np.ndarray, y_train: np.ndarray) -> MultiOutputClassifier:
    """
    Trains a MultiOutput XGBoost classifier to output continuous probabilities for N targets.
    X_train2d should be flat feature vectors, optionally including the LSTM reconstruction error.
    y_train should be shape (num_samples, num_targets)
    """
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='aucpr', # Use PR-AUC for imbalanced data
        use_label_encoder=False,
        random_state=42
    )
    
    # Wrap with MultiOutputClassifier to handle multi-label (Grid, Leak, Burst)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train2d, y_train)
    return model

def predict_risk_score(model: MultiOutputClassifier, X_test2d: np.ndarray) -> np.ndarray:
    """
    Predicts a 0-100 continuous risk score based on the probability of failure for each target.
    Returns: array of shape (n_samples, n_targets) containing probabilities scaled 0-100.
    """
    # predict_proba for MultiOutputClassifier returns a list of arrays (one per target)
    # Each array is shape (n_samples, 2) where col 1 is the positive class probability.
    probs_list = model.predict_proba(X_test2d)
    
    # Extract the positive class probability for each target and stack them horizontally
    positive_probs = [probs[:, 1] for probs in probs_list]
    probs_array = np.column_stack(positive_probs)
    
    # Scale to 0-100
    risk_scores = probs_array * 100
    return risk_scores

def evaluate_model(model: MultiOutputClassifier, X_test2d: np.ndarray, y_test: np.ndarray, target_names: list = None):
    """
    Evaluates the MultiOutput XGBoost model specifically printing PR-AUC (crucial for rare events) per target.
    """
    probs_list = model.predict_proba(X_test2d)
    preds = model.predict(X_test2d)
    
    if target_names is None:
        target_names = [f"Target {i}" for i in range(y_test.shape[1])]
        
    pr_aucs = []
    
    print("\n--- Model Evaluation ---")
    for i, target in enumerate(target_names):
        print(f"\nEvaluating: {target}")
        
        # True bits for target i
        y_true = y_test[:, i]
        # Predicted probs for positive class for target i
        y_score = probs_list[i][:, 1]
        
        pr_auc = average_precision_score(y_true, y_score)
        pr_aucs.append(pr_auc)
        print(f"PR-AUC: {pr_auc:.4f}")
        
        print("Classification Report:")
        print(classification_report(y_true, preds[:, i], zero_division=0))
        
    return pr_aucs
