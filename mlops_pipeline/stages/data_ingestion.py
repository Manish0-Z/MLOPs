import pandas as pd
import numpy as np
import re
import os
import pymysql
from sqlalchemy import create_engine

from paths import load_config, resolve_data_path, get_path

config = load_config()
os.makedirs(get_path("data"), exist_ok=True)


def load_features(features_file):
    with open(features_file, "r", encoding="utf-8") as f:
        content = f.read()
    return re.findall(r'"([^"]+)"', content)


def ingest_data():
    print("=" * 60)
    print("STAGE 1: DATA INGESTION")
    print("=" * 60)

    features_file = resolve_data_path(config["data"]["features_file"])
    features_list = load_features(features_file)
    print(f"Selected features from features.txt: {features_list}")

    accident_cols_map = {
        "Speed_limit": "Speed_limit",
        "Weather_conditions": "Weather_Conditions",
        "Road_type": "Road_Type",
        "Urban_or_rural_area": "Urban_or_Rural_Area",
        "Number_of_Vehicles": "Number_of_Vehicles",
        "Number_of_Casualties": "Number_of_Casualties",
        "Accident_Severity": "Accident_Severity",
    }
    casualty_cols_map = {
        "Age_of_casualty": "Age_of_Casualty",
        "Casualty_class": "Casualty_Class",
    }
    vehicle_cols_map = {
        "Vehicle_type": "Vehicle_Type",
        "Age_of_Vehicle": "Age_of_Vehicle",
        "Engine_Capicity": "Engine_Capacity_(CC)",
        "Age_of_driver": "Age_of_Driver",
    }

    accidents_path = resolve_data_path(config["data"]["accidents"])
    casualties_path = resolve_data_path(config["data"]["casualties"])
    vehicles_path = resolve_data_path(config["data"]["vehicles"])

    accidents = pd.read_csv(
        accidents_path,
        usecols=["Accident_Index"] + list(accident_cols_map.values()),
        low_memory=False,
        encoding="utf-8",
    )
    casualties = pd.read_csv(
        casualties_path,
        usecols=["Accident_Index"] + list(casualty_cols_map.values()),
        low_memory=False,
        encoding="utf-8",
    )
    vehicles = pd.read_csv(
        vehicles_path,
        usecols=["Accident_Index"] + list(vehicle_cols_map.values()),
        low_memory=False,
        encoding="utf-8",
    )

    accidents.rename(columns={v: k for k, v in accident_cols_map.items()}, inplace=True)
    casualties.rename(columns={v: k for k, v in casualty_cols_map.items()}, inplace=True)
    vehicles.rename(columns={v: k for k, v in vehicle_cols_map.items()}, inplace=True)

    cas_agg = casualties.groupby("Accident_Index").agg(
        {
            "Age_of_casualty": "mean",
            "Casualty_class": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        }
    ).reset_index()

    veh_agg = vehicles.groupby("Accident_Index").agg(
        {
            "Vehicle_type": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            "Age_of_Vehicle": "mean",
            "Engine_Capicity": "mean",
            "Age_of_driver": "mean",
        }
    ).reset_index()

    combined = accidents.merge(cas_agg, on="Accident_Index", how="left").merge(
        veh_agg, on="Accident_Index", how="left"
    )

    combined = combined[features_list]
    print(f"Combined dataset shape: {combined.shape}")
    print(f"Columns: {combined.columns.tolist()}")
    print(f"Target distribution:\n{combined[config['target']].value_counts()}")

    data_dir = get_path("data")
    combined.to_csv(get_path("data", "combined_data.csv"), index=False, encoding="utf-8")
    print(f"Saved combined data to {get_path('data', 'combined_data.csv')}")

    db_config = config["database"]
    engine = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}",
        connect_args={"connect_timeout": 3},
    )

    try:
        combined.to_sql(
            config["data"]["combined_table"],
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=10000,
        )
        print(f"Ingested data into MariaDB ColumnStore table: {config['data']['combined_table']}")
    except Exception as e:
        print(f"MariaDB ingestion failed (db may not be running): {e}")
        print("Data saved locally for offline use")

    print("Data ingestion completed successfully!")
    return True


if __name__ == "__main__":
    ingest_data()
