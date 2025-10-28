import joblib
import os

print("🔍 Checking model files...")
files = os.listdir('.')
print("Files in directory:", [f for f in files if '.pkl' in f or '.joblib' in f])

try:
    from feature_engineer import FeatureEngineer
    print("✅ FeatureEngineer imported successfully")
    
    pipeline = joblib.load('bot_detection_pipeline.pkl')
    print("✅ Primary pipeline loaded")
    
    le = joblib.load('label_encoder.pkl')
    print("✅ Label encoder loaded")
    
except Exception as e:
    print(f"❌ Error: {e}")