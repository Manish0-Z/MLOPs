import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from paths import load_config, get_path

config = load_config()


MODEL_MAP = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
}


def train_model():
    print("=" * 60)
    print("STAGE 4: MODEL TRAINING")
    print("=" * 60)

    train_data = pd.read_parquet(get_path("data", "train_data.parquet"))
    test_data = pd.read_parquet(get_path("data", "test_data.parquet"))

    target = config["target"]
    feature_cols = [c for c in train_data.columns if c != target]

    X_train = train_data[feature_cols]
    y_train = train_data[target]
    X_test = test_data[feature_cols]
    y_test = test_data[target]

    print(f"X_train shape: {X_train.shape}, y_train distribution:\n{y_train.value_counts()}")

    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(f"sqlite:///{get_path(mlflow_config['tracking_uri'])}")
    mlflow.set_experiment(mlflow_config["experiment_name"])

    model_type = config["model"]["type"]
    model_params = config["model"]["params"]

    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("n_features", len(feature_cols))

        model_class = MODEL_MAP.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_MAP.keys())}")

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_precision = precision_score(y_test, test_pred, average="weighted", zero_division=0)
        test_recall = recall_score(y_test, test_pred, average="weighted", zero_division=0)
        test_f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        cm = confusion_matrix(y_test, test_pred)
        cm_path = get_path("models", "confusion_matrix.txt")
        with open(cm_path, "w") as f:
            f.write(str(cm))
        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "AccidentSeverityModel")

        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")

        run_info = {
            "run_id": run_id,
            "model_uri": model_uri,
            "metrics": {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
            },
        }
        joblib.dump(run_info, get_path("models", "run_info.pkl"))
        print(f"Saved run info to {get_path('models', 'run_info.pkl')}")

    print("Model training completed successfully!")
    return run_info


if __name__ == "__main__":
    train_model()
