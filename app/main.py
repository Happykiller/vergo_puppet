# app\main.py
import warnings
from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore

from app.inversify import get_inversify
from app.apis.apis import router as model_router  # Import the API routes from the router

# Initialize the FastAPI application
app = FastAPI()
warnings.filterwarnings(
    "ignore",
    message=(
        "torch.utils._pytree._register_pytree_node is deprecated.*"
    ),
)

# Configure allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tools-thomyris.xefi.fr"],  # Allow a specific domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# For init and check
get_inversify()

# Include API routes for model operations
app.include_router(model_router)
