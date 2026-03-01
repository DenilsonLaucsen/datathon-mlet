from fastapi import FastAPI

from app.routes import router

app = FastAPI(title="Datathon ML API")

app.include_router(router)
