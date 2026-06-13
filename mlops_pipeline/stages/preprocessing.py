import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from paths import load_config, get_path

config = load_config()
os.makedirs(get_path("models"), exist_ok=True)


def preprocess_data():
    print("=" * 60)
    print("STAGE 3: DATA PREPROCESSING")
    print("=" * 60)

    data_path = get_path("data", "combined_data.csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    print(f"Loaded data shape: {df.shape}")

    target = config["target"]
    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    for col in numeric_cols:
        strategy = config["preprocessing"]["numeric_impute_strategy"]
        if strategy == "median":
            fill_val = X[col].median()
        else:
            fill_val = X[col].mean()
        X[col] = X[col].fillna(fill_val)

    for col in categorical_cols:
        strategy = config["preprocessing"]["categorical_impute_strategy"]
        if strategy == "most_frequent":
            fill_val = X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown"
        else:
            fill_val = "Unknown"
        X[col] = X[col].fillna(fill_val)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    if config["preprocessing"]["test_size"]:
        test_size = config["preprocessing"]["test_size"]
    else:
        test_size = 0.2
    random_state = config["preprocessing"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")

    train_data = X_train.copy()
    train_data[target] = y_train.values
    test_data = X_test.copy()
    test_data[target] = y_test.values

    data_dir = get_path("data")
    train_data.to_parquet(get_path("data", "train_data.parquet"), index=False)
    test_data.to_parquet(get_path("data", "test_data.parquet"), index=False)

    models_dir = get_path("models")
    preprocessors = {"label_encoders": label_encoders, "scaler": scaler, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}
    joblib.dump(preprocessors, get_path("models", "preprocessors.pkl"))
    print(f"Saved preprocessors to {get_path('models', 'preprocessors.pkl')}")

    print("Preprocessing completed successfully!")
    return True


if __name__ == "__main__":
    preprocess_data()
