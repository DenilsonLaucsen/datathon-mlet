import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils import load_features

TRAIN_DATASET = "data/processed/dataset_academic.csv"
PROD_DATASET = "data/processed/latest_predictions.csv"
FEATURES_PATH = "artifacts/feature_cols.json"
REPORT_PATH = "artifacts/monitoring/drift_report.json"


def calculate_mean_shift(
    train_df: pd.DataFrame, prod_df: pd.DataFrame, features: list[str]
) -> Dict[str, float]:
    """
    Calcula diferença absoluta entre médias das features
    do dataset de treino e dataset de produção.
    """
    drift: Dict[str, float] = {}

    for col in features:
        train_mean = train_df[col].mean()
        prod_mean = prod_df[col].mean()

        drift[col] = float(abs(train_mean - prod_mean))

    return drift


def main() -> None:
    Path("artifacts/monitoring").mkdir(parents=True, exist_ok=True)

    print("Carregando features utilizadas pelo modelo")
    features = load_features(FEATURES_PATH)

    print("Carregando dataset de treino")
    train_df = pd.read_csv(TRAIN_DATASET)

    print("Carregando dataset de produção")
    prod_path = Path(PROD_DATASET)

    if not prod_path.exists():
        raise FileNotFoundError(
            "Arquivo de dados de produção não encontrado: " f"{PROD_DATASET}"
        )

    prod_df = pd.read_csv(prod_path)

    print("Calculando drift")

    drift = calculate_mean_shift(train_df, prod_df, features)

    print("\nDrift detectado (diferença de média):\n")

    with open(REPORT_PATH, "w") as f:
        json.dump(drift, f, indent=2)

    print(json.dumps(drift, indent=2))


if __name__ == "__main__":
    main()
