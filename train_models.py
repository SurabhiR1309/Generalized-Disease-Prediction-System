import pandas as pd
import numpy as np
import os, re, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# === LOAD DATASETS ===
heart = pd.read_csv("datasets/heart.csv")
liver = pd.read_csv("datasets/Liver.csv")
diabetes = pd.read_csv("datasets/PIMA.csv")

# === PREPROCESSING ===
def preprocess(df, target_col, mapping=None):
    df = df.copy()
    df = df.drop(columns=['id'], errors='ignore')
    if mapping:
        for col, m in mapping.items():
            if col in df.columns:
                df[col] = df[col].map(m)
    df = df.replace({'yes':1, 'no':0, 'Male':1, 'Female':0, 'M':1, 'F':0})
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean(numeric_only=True))
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

X_heart, y_heart = preprocess(heart, "target")
X_liver, y_liver = preprocess(liver, "Dataset", {"Gender": {"Male": 1, "Female": 0}})
y_liver = y_liver.map({1:1, 2:0})
X_diabetes, y_diabetes = preprocess(diabetes, "Outcome")

# === TRAINING ===
def train_and_save(X, y, name):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    print(f"{name} accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

    os.makedirs("models", exist_ok=True)
    with open(f"models/{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"models/{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

train_and_save(X_heart, y_heart, "heart")
train_and_save(X_liver, y_liver, "liver")
train_and_save(X_diabetes, y_diabetes, "diabetes")

print("✅ All models trained and saved successfully!")
