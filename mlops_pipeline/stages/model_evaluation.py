import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

from paths import load_config, get_path

config = load_config()


def evaluate_model():
    print("=" * 60)
    print("STAGE 5: MODEL EVALUATION")
    print("=" * 60)

    run_info = joblib.load(get_path("models", "run_info.pkl"))
    test_data = pd.read_parquet(get_path("data", "test_data.parquet"))

    target = config["target"]
    feature_cols = [c for c in test_data.columns if c != target]
    X_test = test_data[feature_cols]
    y_test = test_data[target]

    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(f"sqlite:///{get_path(mlflow_config['tracking_uri'])}")

    model_uri = run_info["model_uri"]
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    evaluation_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }
    joblib.dump(evaluation_results, get_path("models", "evaluation_results.pkl"))
    print(f"Saved evaluation results to {get_path('models', 'evaluation_results.pkl')}")

    with mlflow.start_run(run_id=run_info["run_id"]):
        mlflow.log_metric("eval_accuracy", accuracy)
        mlflow.log_metric("eval_precision", precision)
        mlflow.log_metric("eval_recall", recall)
        mlflow.log_metric("eval_f1", f1)
        mlflow.log_metric("eval_roc_auc", roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted"))

        eval_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        for label, metrics in eval_report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_val in metrics.items():
                    if isinstance(metric_val, (int, float)):
                        mlflow.log_metric(f"{label}_{metric_name}", metric_val)

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='AccidentSeverityModel'")
        latest = max(int(v.version) for v in versions)
        client.set_registered_model_alias("AccidentSeverityModel", "champion", latest)

        print("Registered model alias 'champion' updated in MLflow Model Registry")

    print("Model evaluation completed successfully!")
    return evaluation_results


if __name__ == "__main__":
    evaluate_model()
