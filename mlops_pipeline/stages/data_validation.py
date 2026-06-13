import pandas as pd
import json
import os

from paths import load_config, get_path

config = load_config()


def validate_data():
    print("=" * 60)
    print("STAGE 2: DATA VALIDATION")
    print("=" * 60)

    data_path = get_path("data", "combined_data.csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    print(f"Loaded data shape: {df.shape}")

    ge_dir = get_path("great_expectations")
    os.makedirs(ge_dir, exist_ok=True)

    expectations = [
        {
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": "Accident_Severity"},
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "Accident_Severity"},
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {
                "column": "Accident_Severity",
                "value_set": [1, 2, 3],
            },
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "Speed_limit"},
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "Speed_limit",
                "min_value": 10,
                "max_value": 70,
            },
        },
        {
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": "Weather_conditions"},
        },
        {
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": "Road_type"},
        },
        {
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": "Urban_or_rural_area"},
        },
        {
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": "Vehicle_type"},
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "Number_of_Vehicles"},
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "Number_of_Casualties"},
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "Number_of_Vehicles",
                "min_value": 1,
                "max_value": 100,
            },
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "Number_of_Casualties",
                "min_value": 0,
                "max_value": 100,
            },
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "Age_of_driver"},
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "Age_of_driver",
                "min_value": 1,
                "max_value": 120,
            },
        },
    ]

    suite = {
        "data_asset_type": "Dataset",
        "expectation_suite_name": "accident_data_suite",
        "expectations": expectations,
        "meta": {"great_expectations.__version__": "0.18.0"},
    }

    suite_path = get_path("great_expectations", "expectations.json")
    with open(suite_path, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)
    print(f"Created expectation suite with {len(expectations)} expectations")

    validation_results = {
        "meta": {"suite_name": "accident_data_suite"},
        "results": [],
        "statistics": {"evaluated_expectations": 0, "successful_expectations": 0, "failed_expectations": 0},
    }

    for exp in expectations:
        etype = exp["expectation_type"]
        kwargs = exp["kwargs"]
        result = {"expectation_config": exp, "success": False}

        try:
            if etype == "expect_column_to_exist":
                col = kwargs["column"]
                result["success"] = col in df.columns

            elif etype == "expect_column_values_to_not_be_null":
                col = kwargs["column"]
                null_count = df[col].isnull().sum()
                result["success"] = null_count == 0
                result["result"] = {
                    "observed_value": f"{null_count} null values",
                    "element_count": int(len(df)),
                    "missing_count": int(null_count),
                    "missing_percent": float(round(null_count / len(df) * 100, 2)),
                }

            elif etype == "expect_column_values_to_be_in_set":
                col = kwargs["column"]
                value_set = kwargs["value_set"]
                actual = df[col].dropna().unique()
                result["success"] = all(v in value_set for v in actual)
                result["result"] = {
                    "observed_value": sorted(actual.tolist()),
                    "value_set": value_set,
                }

            elif etype == "expect_column_values_to_be_between":
                col = kwargs["column"]
                min_v = kwargs["min_value"]
                max_v = kwargs["max_value"]
                vals = df[col].dropna()
                out_of_range = ((vals < min_v) | (vals > max_v)).sum()
                result["success"] = out_of_range == 0
                result["result"] = {
                    "observed_min": float(vals.min()),
                    "observed_max": float(vals.max()),
                    "min_value": min_v,
                    "max_value": max_v,
                    "out_of_range_count": int(out_of_range),
                }

        except Exception as e:
            result["success"] = False
            result["exception"] = str(e)

        validation_results["results"].append(result)

        if result["success"]:
            validation_results["statistics"]["successful_expectations"] += 1
        else:
            validation_results["statistics"]["failed_expectations"] += 1
        validation_results["statistics"]["evaluated_expectations"] += 1

    results_path = get_path("great_expectations", "validation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2, default=str)

    stats = validation_results["statistics"]
    print(f"\nValidation Results:")
    print(f"  Evaluated: {stats['evaluated_expectations']}")
    print(f"  Passed:    {stats['successful_expectations']}")
    print(f"  Failed:    {stats['failed_expectations']}")

    if stats["failed_expectations"] > 0:
        print("\nFailed expectations:")
        for r in validation_results["results"]:
            if not r["success"]:
                print(f"  - {r['expectation_config']['expectation_type']}: {r['expectation_config']['kwargs']}")

    all_passed = stats["failed_expectations"] == 0
    print(f"\nData validation {'PASSED' if all_passed else 'FAILED'}!")
    return all_passed


if __name__ == "__main__":
    validate_data()
