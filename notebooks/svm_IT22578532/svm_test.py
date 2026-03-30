import joblib
import numpy as np
import pandas as pd

# Load the trained SVM model
model = joblib.load('notebooks/svm_IT22578532/svm_model.joblib')

# Example: create two test samples (replace with your actual test data)
# The feature order and preprocessing must match your training pipeline!
# Example with 33 features (V1-V28, log_Amount, hour_of_day, scaled_Amount, scaled_Time)
sample1 = {
    'V1': -1.3598, 'V2': -0.0728, 'V3': 2.5364, 'V4': 1.3782,
    'V5': -0.3383, 'V6': -0.4683, 'V7': 0.2076, 'V8': 0.0258,
    'V9': 0.4040, 'V10': 0.2253, 'V11': -0.6295, 'V12': 0.3280,
    'V13': -0.0673, 'V14': -0.2702, 'V15': -0.3984, 'V16': -0.0815,
    'V17': 0.3613, 'V18': 0.1249, 'V19': 0.1417, 'V20': -0.0131,
    'V21': -0.0155, 'V22': 0.0267, 'V23': -0.0108, 'V24': 0.1242,
    'V25': -0.1185, 'V26': 0.0147, 'V27': 0.0097, 'V28': 0.0030,
    'log_Amount': np.log1p(149.62),
    'hour_of_day': (0.0 / 3600) % 24,
    'scaled_Amount': 0.0,  # Replace with your scaler's output
    'scaled_Time': 0.0     # Replace with your scaler's output
}

# A second sample with different values (simulating a different transaction)
sample2 = {
    'V1': 1.1919, 'V2': 0.2661, 'V3': 0.1665, 'V4': 0.4482,
    'V5': 0.0600, 'V6': -0.0824, 'V7': -0.0788, 'V8': 0.0851,
    'V9': -0.2554, 'V10': -0.1661, 'V11': 1.6127, 'V12': 1.0652,
    'V13': 0.4891, 'V14': -0.1437, 'V15': 0.6356, 'V16': 0.4639,
    'V17': -0.1148, 'V18': -0.1834, 'V19': -0.1458, 'V20': -0.0691,
    'V21': -0.2258, 'V22': -0.6387, 'V23': 0.1013, 'V24': -0.2060,
    'V25': 0.5023, 'V26': 0.2194, 'V27': 0.2152, 'V28': -0.0686,
    'log_Amount': np.log1p(2.69),
    'hour_of_day': (50000.0 / 3600) % 24,
    'scaled_Amount': 0.0,  # Replace with your scaler's output
    'scaled_Time': 0.0     # Replace with your scaler's output
}

# A third sample simulating a likely fraud transaction (values inspired by typical fraud patterns)
sample3 = {
    'V1': -2.3122, 'V2': 1.9511, 'V3': -1.6098, 'V4': 3.9979,
    'V5': -0.5222, 'V6': -1.4265, 'V7': -2.5374, 'V8': 1.3916,
    'V9': -2.7700, 'V10': -2.7723, 'V11': 3.2020, 'V12': -2.8999,
    'V13': -0.5952, 'V14': -0.8549, 'V15': -0.8907, 'V16': 0.9370,
    'V17': -0.2180, 'V18': 0.3798, 'V19': -1.5254, 'V20': -1.1197,
    'V21': 0.1751, 'V22': 0.4519, 'V23': -0.2370, 'V24': 0.2654,
    'V25': 0.8000, 'V26': 0.0000, 'V27': 0.0000, 'V28': 0.0000,
    'log_Amount': np.log1p(2125.87),
    'hour_of_day': (86399.0 / 3600) % 24,
    'scaled_Amount': 0.0,  # Replace with your scaler's output
    'scaled_Time': 0.0     # Replace with your scaler's output
}

 
# Convert to DataFrame
X_test = pd.DataFrame([sample1, sample2, sample3, sample4])

# Predict for all samples
probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

for i, (pred, prob) in enumerate(zip(preds, probs), 1):
    print(f"Sample {i} -> Predicted class: {'FRAUD' if pred == 1 else 'NORMAL'} | Fraud probability: {prob:.4f}")