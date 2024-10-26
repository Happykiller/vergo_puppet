from fastapi import FastAPI # type: ignore
from app.apis.apis import router as model_router

app = FastAPI()

# Inclure les routes de l'API
app.include_router(model_router)