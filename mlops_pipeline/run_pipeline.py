"""
Unified runner for the MLOps pipeline.
Run individual stages or the full pipeline sequentially.
Usage:
    python run_pipeline.py --all         # Run full pipeline
    python run_pipeline.py --stage 1     # Run stage 1 only
    python run_pipeline.py --stages 1-5  # Run stages 1 through 5
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGES = {
    1: ("Data Ingestion", "stages.data_ingestion", "ingest_data", None),
    2: ("Data Validation", "stages.data_validation", "validate_data", None),
    3: ("Preprocessing", "stages.preprocessing", "preprocess_data", None),
    4: ("Model Training", "stages.model_training", "train_model", None),
    5: ("Model Evaluation", "stages.model_evaluation", "evaluate_model", None),
    6: ("Model Deployment", "stages.model_deployment", "deploy", {"start_server": False}),
    7: ("Model Monitoring", "stages.model_monitoring", "monitor_model", None),
}


def run_stage(stage_num):
    if stage_num not in STAGES:
        print(f"Error: Stage {stage_num} does not exist. Valid stages: 1-7")
        return False

    name, module_path, func_name, kwargs = STAGES[stage_num]
    print(f"\n{'='*60}")
    print(f"Running Stage {stage_num}: {name}")
    print(f"{'='*60}")

    try:
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        if kwargs:
            result = func(**kwargs)
        else:
            result = func()
        print(f"Stage {stage_num} completed successfully!\n")
        return True
    except Exception as e:
        print(f"Stage {stage_num} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline (stages 1-7)")
    parser.add_argument("--stage", type=int, help="Run a specific stage (1-7)")
    parser.add_argument("--stages", type=str, help="Run a range of stages (e.g., 1-5)")
    args = parser.parse_args()

    stages_to_run = []

    if args.all:
        stages_to_run = list(range(1, 8))
    elif args.stage:
        stages_to_run = [args.stage]
    elif args.stages:
        parts = args.stages.split("-")
        if len(parts) == 2:
            stages_to_run = list(range(int(parts[0]), int(parts[1]) + 1))
    else:
        parser.print_help()
        return

    print("=" * 60)
    print("MLOps Pipeline - Accident Severity Prediction")
    print("=" * 60)

    success = True
    for stage_num in stages_to_run:
        if not run_stage(stage_num):
            print(f"Pipeline halted at Stage {stage_num}")
            success = False
            break

    if success:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("PIPELINE FAILED!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
