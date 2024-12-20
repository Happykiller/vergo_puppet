#app\main.py
from fastapi import FastAPI # type: ignore
from app.apis.apis import router as model_router  # Import the API routes from the router
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI application
app = FastAPI()

# Configuration des origines autorisées
app.add_middleware(
  CORSMiddleware,
  allow_origins=["https://tools-thomyris.xefi.fr"],  # Autoriser le domaine spécifique
  allow_credentials=True,
  allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, PUT, DELETE, etc.)
  allow_headers=["*"],  # Autorise tous les en-têtes
)

# Include API routes for model operations
app.include_router(model_router)
