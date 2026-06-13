import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from paths import load_config, get_path

config = load_config()
os.makedirs(get_path(config["monitoring"]["report_path"]), exist_ok=True)


def generate_evidently_report(reference_data, current_data, feature_cols, target):
    drift_report = {}
    target_drift = {}
    feature_drift = {}
    data_drift = {}

    ref_target = reference_data[target]
    cur_target = current_data[target]

    num_bins = max(1, min(10, len(cur_target.unique())))
    ref_dist = np.histogram(ref_target, bins=num_bins, range=(ref_target.min(), ref_target.max()))[0]
    cur_dist = np.histogram(cur_target, bins=num_bins, range=(ref_target.min(), ref_target.max()))[0]

    ref_dist = ref_dist / ref_dist.sum() if ref_dist.sum() > 0 else ref_dist
    cur_dist = cur_dist / cur_dist.sum() if cur_dist.sum() > 0 else cur_dist

    ps_target = np.sum(np.minimum(ref_dist, cur_dist))
    target_drift["drift_score"] = 1.0 - ps_target
    target_drift["drift_detected"] = target_drift["drift_score"] > 0.1
    target_drift["reference_distribution"] = ref_dist.tolist()
    target_drift["current_distribution"] = cur_dist.tolist()
    target_drift["reference_count"] = int(len(ref_target))
    target_drift["current_count"] = int(len(cur_target))

    drift_report["target_drift"] = target_drift

    for col in feature_cols:
        ref_col = reference_data[col]
        cur_col = current_data[col]

        if ref_col.dtype == "object" or ref_col.nunique() < 20:
            ref_cat = ref_col.value_counts(normalize=True)
            cur_cat = cur_col.value_counts(normalize=True)
            all_cats = set(ref_cat.index) | set(cur_cat.index)
            ref_probs = np.array([ref_cat.get(c, 0) for c in all_cats])
            cur_probs = np.array([cur_cat.get(c, 0) for c in all_cats])
            ps = np.sum(np.minimum(ref_probs, cur_probs))
        else:
            num_bins_f = max(1, min(10, len(ref_col.unique())))
            ref_hist = np.histogram(ref_col.dropna(), bins=num_bins_f)[0]
            cur_hist = np.histogram(cur_col.dropna(), bins=num_bins_f)[0]
            ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_hist = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
            ps = np.sum(np.minimum(ref_hist, cur_hist))

        drift_score = 1.0 - ps
        feature_drift[col] = {
            "drift_score": float(drift_score),
            "drift_detected": drift_score > 0.1,
            "reference_mean": float(ref_col.mean()) if np.issubdtype(ref_col.dtype, np.number) else None,
            "current_mean": float(cur_col.mean()) if np.issubdtype(cur_col.dtype, np.number) else None,
        }

    drift_report["feature_drift"] = feature_drift

    data_drift["n_features"] = len(feature_cols)
    drifted_features = sum(1 for v in feature_drift.values() if v["drift_detected"])
    data_drift["drifted_features"] = drifted_features
    data_drift["drift_ratio"] = drifted_features / len(feature_cols) if feature_cols else 0
    drift_report["data_drift"] = data_drift

    return drift_report


def generate_performance_report(reference_data, current_data, feature_cols, target, model):
    ref_pred = model.predict(reference_data[feature_cols])
    cur_pred = model.predict(current_data[feature_cols])

    performance_report = {}

    ref_metrics = {
        "accuracy": float(accuracy_score(reference_data[target], ref_pred)),
        "precision": float(precision_score(reference_data[target], ref_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(reference_data[target], ref_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(reference_data[target], ref_pred, average="weighted", zero_division=0)),
        "sample_count": len(reference_data),
    }
    cur_metrics = {
        "accuracy": float(accuracy_score(current_data[target], cur_pred)),
        "precision": float(precision_score(current_data[target], cur_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(current_data[target], cur_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(current_data[target], cur_pred, average="weighted", zero_division=0)),
        "sample_count": len(current_data),
    }

    performance_report["reference_metrics"] = ref_metrics
    performance_report["current_metrics"] = cur_metrics

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        performance_report[f"{metric}_degradation"] = ref_metrics[metric] - cur_metrics[metric]

    target_dist_ref = reference_data[target].value_counts(normalize=True).to_dict()
    target_dist_cur = current_data[target].value_counts(normalize=True).to_dict()

    performance_report["target_distribution_reference"] = {str(k): float(v) for k, v in target_dist_ref.items()}
    performance_report["target_distribution_current"] = {str(k): float(v) for k, v in target_dist_cur.items()}

    return performance_report


def monitor_model():
    print("=" * 60)
    print("STAGE 7: MODEL MONITORING")
    print("=" * 60)

    report_dir = get_path(config["monitoring"]["report_path"])
    os.makedirs(report_dir, exist_ok=True)

    train_data = pd.read_parquet(get_path("data", "train_data.parquet"))
    test_data = pd.read_parquet(get_path("data", "test_data.parquet"))

    run_info = joblib.load(get_path("models", "run_info.pkl"))

    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(f"sqlite:///{get_path(mlflow_config['tracking_uri'])}")
    model = mlflow.sklearn.load_model(run_info["model_uri"])

    target = config["target"]
    feature_cols = [c for c in train_data.columns if c != target]

    reference_data = train_data.copy()
    current_data = test_data.copy()

    print(f"Reference data: {len(reference_data)} samples")
    print(f"Current data: {len(current_data)} samples")

    reference_data.to_parquet(get_path(config["monitoring"]["reference_data_path"]), index=False)
    current_data.to_parquet(get_path(config["monitoring"]["current_data_path"]), index=False)

    drift_report = generate_evidently_report(reference_data, current_data, feature_cols, target)
    performance_report = generate_performance_report(reference_data, current_data, feature_cols, target, model)

    monitoring_report = {
        "timestamp": datetime.now().isoformat(),
        "model_uri": run_info["model_uri"],
        "run_id": run_info["run_id"],
        "drift_report": drift_report,
        "performance_report": performance_report,
    }

    report_file = get_path(
        config["monitoring"]["report_path"],
        f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(monitoring_report, f, indent=2, default=str)
    print(f"Monitoring report saved to {report_file}")

    print(f"\nDrift Analysis:")
    print(f"  Target drift: {'DETECTED' if drift_report['target_drift']['drift_detected'] else 'Not detected'} "
          f"(score: {drift_report['target_drift']['drift_score']:.4f})")
    print(f"  Feature drift: {drift_report['data_drift']['drifted_features']}/{drift_report['data_drift']['n_features']} "
          f"features drifted (ratio: {drift_report['data_drift']['drift_ratio']:.2%})")

    drifted = [col for col, info in drift_report["feature_drift"].items() if info["drift_detected"]]
    if drifted:
        print(f"  Drifted features: {drifted}")

    print(f"\nPerformance Comparison:")
    ref_metrics = performance_report["reference_metrics"]
    cur_metrics = performance_report["current_metrics"]
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        change = cur_metrics[metric] - ref_metrics[metric]
        arrow = "+" if change > 0 else ""
        print(f"  {metric}: {ref_metrics[metric]:.4f} -> {cur_metrics[metric]:.4f} ({arrow}{change:+.4f})")

    needs_retraining = (
        drift_report["target_drift"]["drift_detected"]
        or drift_report["data_drift"]["drift_ratio"] > 0.3
    )

    if needs_retraining:
        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": "retraining_needed",
            "severity": "high",
            "message": "Model drift detected — retraining pipeline should be triggered",
            "details": {
                "target_drift_score": drift_report["target_drift"]["drift_score"],
                "feature_drift_ratio": drift_report["data_drift"]["drift_ratio"],
                "accuracy_degradation": performance_report.get("accuracy_degradation", 0),
            },
        }
        alert_file = get_path(config["monitoring"]["report_path"], "alert_retraining.json")
        with open(alert_file, "w", encoding="utf-8") as f:
            json.dump(alert, f, indent=2)
        print(f"\nALERT: Model drift detected! Retraining alert saved to {alert_file}")
    else:
        print("\nModel performance is stable. No retraining needed.")

    with mlflow.start_run(run_id=run_info["run_id"]):
        mlflow.log_metric("drift_target_score", drift_report["target_drift"]["drift_score"])
        mlflow.log_metric("drift_feature_ratio", drift_report["data_drift"]["drift_ratio"])
        mlflow.log_metric("drifted_features", drift_report["data_drift"]["drifted_features"])
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            mlflow.log_metric(f"monitoring_{metric}_degradation", performance_report.get(f"{metric}_degradation", 0))
        mlflow.log_artifact(report_file)

    print("Model monitoring completed successfully!")
    return monitoring_report


if __name__ == "__main__":
    monitor_model()
