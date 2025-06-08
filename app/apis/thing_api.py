# app/apis/thing_api.py
from fastapi import APIRouter, Depends, HTTPException

from app.inversify import get_inversify
from app.apis.deps import verify_access_token
from app.usecases.thing.usecase_get_things import get_things_usecase
from app.usecases.thing.usecase_store_thing import store_thing_usecase

thing_router = APIRouter()

@thing_router.post("/thing")
async def store_thing_api(item: dict, model_name: str, payload: dict = Depends(verify_access_token)):
    """
    Stores a new Thing with an embedding.
    """
    try:
        result = store_thing_usecase(item, model_name, get_inversify())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@thing_router.get("/thing")
async def get_things_api(ids: list[str] = None, payload: dict = Depends(verify_access_token)):
    """
    Retrieves Things by ID list or all if no ids provided.
    """
    try:
        result = get_things_usecase(ids, get_inversify())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
