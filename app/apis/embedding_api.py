# app/apis/embedding_api.py
from pydantic import BaseModel
from app.apis.common import load_json_file
from fastapi import APIRouter, HTTPException, Depends, Body

from app.apis.apis import FILES_DIR
from app.inversify import get_inversify
from app.apis.apis import verify_access_token
from app.usecases.embedding.usecase_train_embedding import train_embedding_usecase
from app.usecases.embedding.usecase_create_embedding import create_embedding_usecase
from app.usecases.embedding.usecase_encode_embedding import encode_embedding_usecase
from app.usecases.embedding.usecase_similarity_embedding import similarity_embedding_usecase

embedding_router = APIRouter()

class EmbeddingTrainRequest(BaseModel):
    model_name: str
    vocab_path: str
    trainset_path: str

@embedding_router.post("/embedding/create")
async def train_embedding_api(
    params: EmbeddingTrainRequest,
    payload: dict = Depends(verify_access_token)
):
    """
    Create the embedding model on preprocessed dataset.
    """
    try:
        trainset = load_json_file(FILES_DIR / params.trainset_path)
        vocab = load_json_file(FILES_DIR / params.vocab_path)

        result = create_embedding_usecase(params.model_name, vocab, trainset, get_inversify())
        return {"status": "ok", "detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@embedding_router.post("/embedding/train")
async def train_embedding_api(
    params: EmbeddingTrainRequest,
    payload: dict = Depends(verify_access_token)
):
    """
    Train the embedding model on preprocessed dataset.
    """
    try:
        trainset = load_json_file(FILES_DIR / params.trainset_path)
        vocab = load_json_file(FILES_DIR / params.vocab_path)

        result = train_embedding_usecase(params.model_name, vocab, trainset, get_inversify())
        return {"status": "ok", "detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@embedding_router.post("/embedding/encode")
async def encode_embedding_api(
    model_name: str = Body(..., embed=True),
    sentence: str = Body(..., embed=True),
    payload: dict = Depends(verify_access_token)
):
    """
    Encode a single sentence to its embedding vector.
    """
    try:
        embedding = encode_embedding_usecase(model_name, sentence, get_inversify())
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

@embedding_router.post("/embedding/similarity")
async def similarity_embedding_api(
    model_name: str = Body(..., embed=True),
    sentence1: str = Body(..., embed=True),
    sentence2: str = Body(..., embed=True),
    payload: dict = Depends(verify_access_token)
):
    """
    Compute cosine similarity between embeddings of two sentences.
    """
    try:
        score = similarity_embedding_usecase(model_name, sentence1, sentence2, get_inversify())
        return {"similarity": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")

