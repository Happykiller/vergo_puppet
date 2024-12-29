# app\usecases\gru\usecase_search_multi_brut_gru.py
from typing import List, NamedTuple
from fastapi import HTTPException  # type: ignore

from app.inversify import Inversify
from app.neural_network.nn_gru import predict
from app.usecases.usecase_tokenize import usecase_tokenize
from app.usecases.gru.usecase_commons_gru import process_input
from app.apis.models.tokenize_model_data import ModelTokenizeData

class SearchMultiBrutGRUUsecaseDto(NamedTuple):
    name: str
    documents: List[ModelTokenizeData]
    inversify: Inversify

def search_multi_brut_model_gru(dto: SearchMultiBrutGRUUsecaseDto):
    result = []
    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    # Retrieve model data from memory
    model_data = bdd.get_model(dto.name)
    
    # Check if the model data is found
    if model_data is None or not model_data:
        # Raise a 404 error if the model is not found
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Extract the neural network model from the data
    nn_model = model_data.get("nn_model", None)
    if nn_model is None:
        # Raise a 400 error if the model is not trained
        raise HTTPException(status_code=400, detail="Model not trained")
    
    # Retrieve word-to-index and index-to-category mappings
    word2idx = model_data.get("word2idx", None)
    idx2category = model_data.get("idx2category", None)
    if word2idx is None or idx2category is None:
        # Raise a 400 error if essential model data is incomplete
        raise HTTPException(status_code=400, detail="Model data incomplete")
    
    documents_tokenized = usecase_tokenize(dto.documents)

    for index, data in enumerate(documents_tokenized):
        # Process the input sequence using the word-to-index mapping
        input = process_input(data['tokens'], word2idx)

        # Make a prediction using the GRU model
        predicted_idx = predict(nn_model, input)

        # Map the predicted index to the corresponding category
        predicted_category = idx2category[predicted_idx]
    
        result.append({
            'incidentId': dto.documents[index].incidentId,
            'description': dto.documents[index].description,
            'tokens': data['tokens'],
            'predicted_category': predicted_category
        })

    return result
