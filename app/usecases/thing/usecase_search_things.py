# app\usecases\thing\usecase_search_things.py
import torch
import traceback
import numpy as np

from app.services.logger import logger
from app.usecases.get_model import get_model_usecase
from app.usecases.thing.usecase_get_things import get_things_usecase
from app.usecases.thing.usecase_store_thing import encode_text_with_model
from app.usecases.embedding.usecase_similarity_embedding import cosine_similarity

def search_things_usecase(model_name: str, collection_name: str, sentence: str, top_k: int, inversify):
    try:
        model = get_model_usecase(model_name, inversify)
        if not model or not model.nn_model:
            raise ValueError("Model not loaded or incomplete")

        # Tokenize + encode sentence
        vector = encode_text_with_model(model, sentence)

        # Récupère les choses dans la collection
        things = get_things_usecase(collection_name, ids=None, inversify=inversify)
        if not things:
            return {
                "search": sentence,
                "matchs": []
            }

        # Calcule les similarités
        results = []
        for t in things:
            score = cosine_similarity(vector, t.vector)
            results.append({
                "id": t.id,
                "text": t.text,
                "score": score
            })

        # 4. Trie et limite
        results.sort(key=lambda x: x["score"], reverse=True)
        top_matches = results[:top_k]

        # 5. Structure finale
        return {
            "search": sentence,
            "matchs": top_matches
        }

    except Exception as e:
        logger.error(f"[search_things_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[search_things_usecase] {str(e)}")
