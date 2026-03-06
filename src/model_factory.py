from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_model(model_name: str):
    if model_name == "logreg":
        return LogisticRegression(random_state=42, max_iter=1000)

    if model_name == "rf":
        return RandomForestClassifier(random_state=42)

    if model_name == "gb":
        return GradientBoostingClassifier(random_state=42)

    if model_name == "xgb":
        return XGBClassifier(
            random_state=42, eval_metric="logloss", use_label_encoder=False
        )

    if model_name == "lgbm":
        return LGBMClassifier(random_state=42)

    raise ValueError(f"Modelo não suportado: {model_name}")
