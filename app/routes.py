from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
def predict(payload: dict):
    return {"prediction": None, "warning": "modelo ainda não treinado"}
