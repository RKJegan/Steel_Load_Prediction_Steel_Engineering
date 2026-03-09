import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop_duplicates()

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

def encode_data(df, save_encoder=False, encoder_path="models/label_encoder.pkl"):
    if "steel_grade" in df.columns:
        le = LabelEncoder()
        df["steel_grade"] = le.fit_transform(df["steel_grade"])

        if save_encoder:
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            joblib.dump(le, encoder_path)

    return df
