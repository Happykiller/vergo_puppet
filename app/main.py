#app\main.py
from fastapi import FastAPI # type: ignore
from app.apis.apis import router as model_router  # Import the API routes from the router

# Initialize the FastAPI application
app = FastAPI()

# Include API routes for model operations
app.include_router(model_router)
