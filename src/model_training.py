import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

def train_models(df):
    X = df.drop("failure_load", axis=1)
    y = df["failure_load"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        print(name, "R2:", score)

        if score > best_score:
            best_score = score
            best_model = model

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, os.path.join("models", "best_model.pkl"))

    return best_model, X_test, y_test
