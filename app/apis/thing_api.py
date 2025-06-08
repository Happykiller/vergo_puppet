# app/apis/thing_api.py
from pydantic import BaseModel
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException

from app.inversify import get_inversify
from app.apis.deps import verify_access_token
from app.usecases.thing.usecase_get_things import get_things_usecase
from app.usecases.thing.usecase_store_thing import store_thing_usecase
from app.usecases.thing.usecase_search_things import search_things_usecase

thing_router = APIRouter()

class StoreThingInput(BaseModel):
    model_encode_name: str
    collection_name: str
    id: str
    data: dict
    
class SearchThingInput(BaseModel):
    collection_name: str
    ids: Optional[list[str]] = None
    
class ThingSearchInput(BaseModel):
    model_encode_name: str
    collection_name: str
    sentence: str
    top_k: Optional[int] = 10

@thing_router.post("/thing")
async def store_thing_api(body: StoreThingInput, payload: dict = Depends(verify_access_token)):
    """Store a new thing and its embedding."""
    try:
        result = store_thing_usecase(
            model_name=body.model_encode_name,
            collection_name=body.collection_name,
            thing_id=body.id,
            data=body.data,
            inversify=get_inversify()
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@thing_router.post("/thing/list")
async def get_things_api(body: SearchThingInput, payload: dict = Depends(verify_access_token)):
    """Retrieve things from a collection."""
    try:
        result = get_things_usecase(body.collection_name, ids=body.ids, inversify=get_inversify())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@thing_router.post("/thing/search")
async def search_thing_api(body: ThingSearchInput, payload: dict = Depends(verify_access_token)):
    """Search for similar things based on a sentence."""
    try:
        result = search_things_usecase(
            model_name=body.model_encode_name,
            collection_name=body.collection_name,
            sentence=body.sentence,
            top_k=body.top_k,
            inversify=get_inversify()
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))