# app/apis/embedding_api.py
from threading import Thread
from pydantic import BaseModel # type: ignore
from fastapi import APIRouter, HTTPException, Depends, Body # type: ignore

from app.apis.apis import FILES_DIR
from app.inversify import get_inversify
from app.apis.common import load_json_file
from app.apis.apis import verify_access_token
from app.services.logger import logger
from app.usecases.embedding.usecase_train_embedding import train_embedding_usecase
from app.usecases.embedding.usecase_create_embedding import create_embedding_usecase
from app.usecases.embedding.usecase_encode_embedding import encode_embedding_usecase
from app.usecases.embedding.usecase_similarity_embedding import similarity_embedding_usecase
from app.usecases.embedding.usecase_mesure_embedding import (
    MesureEmbeddingUsecaseDto,
    mesure_embedding,
)

embedding_router = APIRouter()

class EmbeddingTrainRequest(BaseModel):
    model_name: str
    vocab_path: str
    trainset_path: str


class EmbeddingMesureRequest(BaseModel):
    model_name: str
    test_path: str


def _create_embedding_thread(model_name: str, vocab: dict, trainset: list[dict]) -> None:
    """Run create_embedding_usecase in a dedicated thread."""
    try:
        create_embedding_usecase(model_name, vocab, trainset, get_inversify())
    except Exception as exc:
        logger.error(f"[create_embedding_thread] {str(exc)}")


def _train_embedding_thread(model_name: str, vocab: dict, trainset: list[dict]) -> None:
    """Run train_embedding_usecase in a dedicated thread."""
    try:
        train_embedding_usecase(model_name, vocab, trainset, get_inversify())
    except Exception as exc:
        logger.error(f"[train_embedding_thread] {str(exc)}")

@embedding_router.post("/embedding/create")
async def create_embedding_api(
    params: EmbeddingTrainRequest,
    payload: dict = Depends(verify_access_token)
):
    """
    Create the embedding model on preprocessed dataset.
    """
    try:
        trainset = load_json_file(FILES_DIR / params.trainset_path)
        vocab = load_json_file(FILES_DIR / params.vocab_path)
        
        if not isinstance(vocab, dict):
            raise HTTPException(status_code=400, detail="Vocabulary must be a JSON object (dict).")
        
        if not (isinstance(trainset, list) and all(isinstance(item, dict) for item in trainset)):
            raise HTTPException(
                status_code=400,
                detail="Trainset must be a JSON array of objects (list of dicts)."
            )

        Thread(
            target=_create_embedding_thread,
            args=(params.model_name, vocab, trainset),
            daemon=True,
        ).start()

        return {"status": "started"}
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
        
        if not isinstance(vocab, dict):
            raise HTTPException(status_code=400, detail="Vocabulary must be a JSON object (dict).")
        
        if not (isinstance(trainset, list) and all(isinstance(item, dict) for item in trainset)):
            raise HTTPException(
                status_code=400,
                detail="Trainset must be a JSON array of objects (list of dicts)."
            )

        Thread(
            target=_train_embedding_thread,
            args=(params.model_name, vocab, trainset),
            daemon=True,
        ).start()

        return {"status": "started"}
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


@embedding_router.post("/embedding/mesure")
async def mesure_embedding_api(
    params: EmbeddingMesureRequest,
    payload: dict = Depends(verify_access_token),
):
    """Measure the embedding model performance on a test dataset."""
    try:
        test_data = load_json_file(FILES_DIR / params.test_path)
        
        if not (isinstance(test_data, list) and all(isinstance(item, dict) for item in test_data)):
            raise HTTPException(
                status_code=400,
                detail="Testset must be a JSON array of objects (list of dicts)."
            )
        
        result = mesure_embedding(
            MesureEmbeddingUsecaseDto(
                name=params.model_name,
                test_data=test_data,
                inversify=get_inversify(),
            )
        )
        return {"status": "ok", "detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mesure failed: {str(e)}")

