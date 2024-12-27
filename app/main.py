# app\main.py
from fastapi import FastAPI  # type: ignore
from app.apis.apis import router as model_router  # Import the API routes from the router
from fastapi.middleware.cors import CORSMiddleware  # type: ignore

# Initialize the FastAPI application
app = FastAPI()

# Configure allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tools-thomyris.xefi.fr"],  # Allow a specific domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include API routes for model operations
app.include_router(model_router)
