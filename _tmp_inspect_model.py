import joblib
from pathlib import Path

model = joblib.load(Path(r'C:/Users/kryst/OneDrive/Documents/skola/eeg_app/models/model_latest.joblib'))
print(type(model))
print(model)
if hasattr(model, 'named_steps'):
    print('steps', list(model.named_steps.keys()))
    scaler = model.named_steps.get('scaler')
    clf = model.named_steps.get('clf')
    print('scaler_features', getattr(scaler, 'n_features_in_', None))
    print('clf_features', getattr(clf, 'n_features_in_', None))
    print('model_features', getattr(model, 'n_features_in_', None))
