import joblib
import pandas as pd

def try_load(path):
    try:
        obj = joblib.load(path)
        print(f"✅ Loaded: {path}")
        return obj
    except Exception as e:
        print(f"❌ Failed to load: {path}")
        print("Error:", e)
        return None

print("🔍 Checking model files in this folder...\n")

xgb = try_load("xgb_model.joblib")
lgb = try_load("lgb_model.joblib")
scaler = try_load("scaler.joblib")

# test if prediction works
if xgb is not None and lgb is not None and scaler is not None:
    sample = pd.DataFrame([{
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 80,
        'BMI': 28.0,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 35,
        'BMI_Age': 28.0 * 35,
        'Glucose_Age': 120 * 35,
        'Glucose_BMI': 120 * 28.0,
        'Age_Group': 2
    }])
    X_scaled = scaler.transform(sample)
    prob_xgb = xgb.predict_proba(X_scaled)[:, 1][0]
    prob_lgb = lgb.predict_proba(X_scaled)[:, 1][0]
    print(f"\n✅ Prediction test passed — XGB: {prob_xgb:.3f}, LGB: {prob_lgb:.3f}")
else:
    print("\n⚠️ Some models failed to load — retraining is needed.")
