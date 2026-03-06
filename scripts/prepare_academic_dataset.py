import json
import os

import pandas as pd

INPUT = "data/processed/dataset_consolidado.csv"
OUT_CSV = "data/processed/dataset_academic.csv"
OUT_FEATURES = "artifacts/feature_cols.json"

academic_cols = ["Matematica", "Portugues", "Ingles", "IDA", "Cg", "Cf", "Ct"]
target = "defasado_bin"

os.makedirs("artifacts", exist_ok=True)
df = pd.read_csv(INPUT)

if target not in df.columns:
    if "Defasagem" not in df.columns:
        raise ValueError("Defasagem não encontrada no CSV original.")
    df[target] = (df["Defasagem"] < 0).astype(int)

missing = [c for c in academic_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colunas acadêmicas ausentes: {missing}")

df_out = df[academic_cols + [target]].copy()
df_out.to_csv(OUT_CSV, index=False)

with open(OUT_FEATURES, "w") as f:
    json.dump({"features": academic_cols}, f, indent=2)

print(f"Saved {OUT_CSV} and feature list -> {OUT_FEATURES}")
