import json
import os


def pytest_sessionstart(session):
    os.makedirs("artifacts", exist_ok=True)

    feature_cols = {
        "features": ["Matematica", "Portugues", "Ingles", "IDA", "Cg", "Cf", "Ct"]
    }

    with open("artifacts/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
