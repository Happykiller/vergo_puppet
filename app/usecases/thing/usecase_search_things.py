# app\usecases\thing\usecase_search_things.py
import traceback

from app.services.logger import logger
from app.usecases.get_model import get_model_usecase
from app.usecases.thing.usecase_get_things import get_things_usecase
from app.usecases.embedding.usecase_similarity_embedding import cosine_similarity
from app.usecases.embedding.usecase_encode_embedding import encode_embedding_usecase

def search_things_usecase(model_name: str, collection_name: str, sentence: str, top_k: int, inversify):
    """Search for the most similar things within a collection."""
    try:
        model = get_model_usecase(model_name, inversify)
        if not model:
            raise ValueError("Model not loaded or incomplete")

        # Tokenize and encode the query sentence
        vector = encode_embedding_usecase(model_name, sentence, inversify)

        # Retrieve things stored in the requested collection
        things = get_things_usecase(collection_name, ids=None, inversify=inversify)
        if not things:
            return {
                "search": sentence,
                "matchs": []
            }

        # Compute cosine similarities
        results = []
        for t in things:
            score = cosine_similarity(vector, t.vector)
            results.append({
                "id": t.id,
                "text": t.text,
                "score": score
            })

        # Sort by score and keep only top K
        results.sort(key=lambda x: x["score"], reverse=True)
        top_matches = results[:top_k]

        # Final response structure
        return {
            "search": sentence,
            "matchs": top_matches
        }

    except Exception as e:
        logger.error(f"[search_things_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[search_things_usecase] {str(e)}")
