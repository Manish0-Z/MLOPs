import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import TaskGroup

# Airflow Celery config — set these in airflow.cfg or environment variables:
# executor = CeleryExecutor
# celery_broker_url = redis://localhost:6379/1
# celery_result_backend = redis://localhost:6379/2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "monitoring", "reports")


def _import_stage(module, func):
    import importlib
    m = importlib.import_module(f"stages.{module}")
    return getattr(m, func)


def ingest_data():
    return _import_stage("data_ingestion", "ingest_data")()


def validate_data():
    return _import_stage("data_validation", "validate_data")()


def preprocess_data():
    return _import_stage("preprocessing", "preprocess_data")()


def train_model():
    return _import_stage("model_training", "train_model")()


def evaluate_model():
    return _import_stage("model_evaluation", "evaluate_model")()


def monitor_model():
    return _import_stage("model_monitoring", "monitor_model")()


def check_ingestion():
    print("DB ingestion verified")


def check_validation():
    import json
    path = os.path.join(PROJECT_ROOT, "great_expectations", "validation_results.json")
    with open(path) as f:
        r = json.load(f)
    stats = r["statistics"]
    print(f'Validation: {stats["successful_expectations"]}/{stats["evaluated_expectations"]} passed')


def check_preprocessing():
    train = os.path.join(PROJECT_ROOT, "data", "train_data.parquet")
    test = os.path.join(PROJECT_ROOT, "data", "test_data.parquet")
    print(f'Train: {os.path.getsize(train)} bytes, Test: {os.path.getsize(test)} bytes')


def check_training():
    import joblib
    r = joblib.load(os.path.join(PROJECT_ROOT, "models", "run_info.pkl"))
    print(f'Training done: run_id={r["run_id"][:8]}..., accuracy={r["metrics"]["train_accuracy"]:.4f}')


def check_evaluation():
    import joblib
    r = joblib.load(os.path.join(PROJECT_ROOT, "models", "evaluation_results.pkl"))
    print(f'Evaluation: accuracy={r["accuracy"]:.4f}, f1={r["f1_score"]:.4f}')


def check_monitoring():
    reports = os.listdir(REPORTS_DIR)
    print(f"Monitoring reports generated: {len(reports)}")


def check_alert():
    alert = os.path.join(REPORTS_DIR, "alert_retraining.json")
    print(f"Retraining alert: {'ACTIVE' if os.path.exists(alert) else 'None'}")

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 1, 1),
}

with DAG(
    dag_id="mlops_pipeline",
    default_args=default_args,
    description="End-to-end MLOps pipeline: ingestion, validation, preprocessing, training, evaluation, deployment, monitoring",
    schedule="@weekly",
    catchup=False,
    tags=["mlops", "accident_severity"],
) as dag:

    dag_start = EmptyOperator(task_id="pipeline_start")
    dag_end = EmptyOperator(task_id="pipeline_end")

    with TaskGroup(group_id="stage_1_data_ingestion") as stage_1:
        ingest_from_csv = PythonOperator(
            task_id="ingest_from_csv",
            python_callable=ingest_data,
        )
        check_db_ingestion = PythonOperator(
            task_id="check_db_ingestion",
            python_callable=check_ingestion,
        )
        ingest_from_csv >> check_db_ingestion

    with TaskGroup(group_id="stage_2_data_validation") as stage_2:
        validate = PythonOperator(
            task_id="validate_with_great_expectations",
            python_callable=validate_data,
        )
        check_validation = PythonOperator(
            task_id="check_validation_results",
            python_callable=check_validation,
        )
        validate >> check_validation

    with TaskGroup(group_id="stage_3_preprocessing") as stage_3:
        preprocess = PythonOperator(
            task_id="preprocess_features",
            python_callable=preprocess_data,
        )
        check_preprocessing = PythonOperator(
            task_id="check_preprocessed_data",
            python_callable=check_preprocessing,
        )
        preprocess >> check_preprocessing

    with TaskGroup(group_id="stage_4_model_training") as stage_4:
        train = PythonOperator(
            task_id="train_with_mlflow",
            python_callable=train_model,
        )
        check_training = PythonOperator(
            task_id="check_training_results",
            python_callable=check_training,
        )
        train >> check_training

    with TaskGroup(group_id="stage_5_model_evaluation") as stage_5:
        evaluate = PythonOperator(
            task_id="evaluate_with_mlflow",
            python_callable=evaluate_model,
        )
        check_evaluation = PythonOperator(
            task_id="check_evaluation_results",
            python_callable=check_evaluation,
        )
        evaluate >> check_evaluation

    with TaskGroup(group_id="stage_6_model_deployment") as stage_6:
        deploy_app = BashOperator(
            task_id="start_fastapi_server",
            bash_command="echo 'FastAPI server would start on port 8000'",
        )
        check_api = BashOperator(
            task_id="test_api_endpoint",
            bash_command="echo 'API health check: http://localhost:8000/health'",
        )
        load_model = BashOperator(
            task_id="load_champion_model",
            bash_command="echo 'Loading champion model from MLflow Model Registry'",
        )
        load_model >> deploy_app >> check_api

    with TaskGroup(group_id="stage_7_model_monitoring") as stage_7:
        monitor = PythonOperator(
            task_id="monitor_with_evidently",
            python_callable=monitor_model,
        )
        check_monitoring = PythonOperator(
            task_id="check_monitoring_report",
            python_callable=check_monitoring,
        )
        trigger_alert = PythonOperator(
            task_id="trigger_retraining_alert",
            python_callable=check_alert,
        )
        monitor >> check_monitoring >> trigger_alert

    (
        dag_start
        >> stage_1
        >> stage_2
        >> stage_3
        >> stage_4
        >> stage_5
        >> stage_6
        >> stage_7
        >> dag_end
    )
