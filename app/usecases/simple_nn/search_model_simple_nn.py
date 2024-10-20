import joblib
from app.repositories.memory import get_model
from fastapi import HTTPException # type: ignore
from app.machine_learning.neural_network_simple import predict
from app.apis.models.simple_nn_search_data import SimpleNNSearchData
from app.usecases.simple_nn.simple_nn_commons import process_input_data

def search_model_simple_nn(name: str, search: SimpleNNSearchData):
    model = get_model(name)
    
    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    nn_model = model.get("nn_model", None)
    if nn_model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Charger l'encodeur, le scaler et les indices
    encoder_filename = model.get("encoder_filename")
    scaler_filename = model.get("scaler_filename")
    indices_filename = model.get("indices_filename")
    if not encoder_filename or not scaler_filename or not indices_filename:
        raise HTTPException(status_code=400, detail="Missing encoder, scaler, or indices in the model")
    
    encoder = joblib.load(encoder_filename)
    scaler = joblib.load(scaler_filename)
    indices_info = joblib.load(indices_filename)
    categorical_indices = indices_info["categorical_indices"]
    numerical_indices = indices_info["numerical_indices"]
    
    # Récupérer les paramètres de normalisation des targets
    targets_mean = model.get("targets_mean")
    targets_std = model.get("targets_std")
    if targets_mean is None or targets_std is None:
        raise HTTPException(status_code=400, detail="Missing normalization parameters in the model")
    
    # Préparer les données d'entrée
    input_data = [
        search.type,
        search.surface,
        search.pieces,
        search.floor,
        search.parking,
        search.balcon,
        search.ascenseur,
        search.orientation,
        search.transports,
        search.neighborhood
    ]
    
    # Transformer les données d'entrée
    input_processed = process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices)
    
    # Prédiction
    predicted = predict(nn_model, input_processed, targets_mean, targets_std)
    
    return {"predicted_price": predicted}
