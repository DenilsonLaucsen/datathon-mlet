from sklearn.pipeline import Pipeline

from src.model_factory import get_model
from src.preprocessing import build_preprocessor


def build_pipeline(features: list[str], model_name: str):
    preprocessor = build_preprocessor(features)

    model = get_model(model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
