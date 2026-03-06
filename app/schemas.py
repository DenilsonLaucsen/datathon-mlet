from pydantic import BaseModel


class PredictionRequest(BaseModel):
    Matematica: float
    Portugues: float
    Ingles: float
    IDA: float
    Cg: float
    Cf: float
    Ct: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
