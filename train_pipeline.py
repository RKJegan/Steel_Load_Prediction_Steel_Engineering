from src.data_processing import load_data, clean_data, encode_data
from src.feature_engineering import create_features
from src.model_training import train_models
from src.evaluate_model import evaluate

data_path = "data/steel_dataset.csv"

df = load_data(data_path)
df = clean_data(df)
df = encode_data(df, save_encoder=True, encoder_path="models/label_encoder.pkl")
df = create_features(df)

model, X_test, y_test = train_models(df)
evaluate(model, X_test, y_test)

print("Saved model to models/best_model.pkl")
print("Saved label encoder to models/label_encoder.pkl")
