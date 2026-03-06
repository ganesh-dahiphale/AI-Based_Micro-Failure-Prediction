def predict_rul(risk_score: float) -> str:
    """
    Estimates the Remaining Useful Life (RUL) based on the current XGBoost risk score.
    Since this is simulated without true continuous labels, we use a calibrated exponential decay heuristic.
    
    Risk 0-20: 30+ days
    Risk 20-50: 10 - 30 days
    Risk 50-80: 2 - 10 days
    Risk 80-95: 12 - 48 hours
    Risk > 95: < 12 hours
    """
    
    if risk_score > 95:
        # Convert to hours
        hours_left = max(0.5, 12.0 - ((risk_score - 95) * 2))
        return f"{hours_left:.1f} hours"
        
    elif risk_score > 80:
        # Range: 12 to 48 hours
        ratio = (95 - risk_score) / 15.0  # 0 to 1 (1 means closer to 80)
        hours_left = 12.0 + (ratio * 36.0)
        return f"{hours_left:.1f} hours"
        
    elif risk_score > 50:
        # Range: 2 to 10 days
        ratio = (80 - risk_score) / 30.0
        days_left = 2.0 + (ratio * 8.0)
        return f"{days_left:.1f} days"
        
    elif risk_score > 20:
        # Range 10 to 30 days
        ratio = (50 - risk_score) / 30.0
        days_left = 10.0 + (ratio * 20.0)
        return f"{days_left:.1f} days"
        
    else:
        # Base healthy
        return "> 30 days"
