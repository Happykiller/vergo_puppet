import joblib
from typing import List
from app.services.logger import logger
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import get_model, update_model
from app.usecases.simple_nn.simple_nn_commons import transform_data
from app.machine_learning.neural_network_simple import train_model_nn
from app.apis.models.simple_nn_training_data import SimpleNNTrainingData

def train_model_simple_nn(name: str, training_data: List[SimpleNNTrainingData]):
    """
    Entraîne le modèle avec des données d'entraînement fournies.
    :param name: Nom du modèle
    :param training_data: Liste des données d'entraînement
    """
    model = get_model(name)
    
    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if training_data is None or len(training_data) == 0:
        raise HTTPException(status_code=400, detail="No training data provided or training data is empty")
    
    logger.info(f"Type de machine learning utilisé pour l'entraînement SimpleNN")
    
    # Transformation des données
    features_processed, targets_standardized, encoder, scaler, targets_mean, targets_std, categorical_indices, numerical_indices = transform_data(training_data)
    
    # Récupérer la taille des features
    input_size = features_processed.shape[1]
    
    # Entraîner le réseau de neurones
    nn_model, losses = train_model_nn(features_processed, targets_standardized, input_size)
    
    # Enregistrer l'encodeur, le scaler et les paramètres de normalisation des targets
    encoder_filename = f"{name}_encoder.joblib"
    scaler_filename = f"{name}_scaler.joblib"
    joblib.dump(encoder, encoder_filename)
    joblib.dump(scaler, scaler_filename)
    
    # Enregistrer les indices
    indices_info = {
        "categorical_indices": categorical_indices,
        "numerical_indices": numerical_indices
    }
    indices_filename = f"{name}_indices.joblib"
    joblib.dump(indices_info, indices_filename)
    
    # Enregistrer le modèle de réseau de neurones entraîné
    update_model(name, {
        "nn_model": nn_model,
        "encoder_filename": encoder_filename,
        "scaler_filename": scaler_filename,
        "indices_filename": indices_filename,
        "targets_mean": targets_mean,
        "targets_std": targets_std
    })
    
    return {"status": "training completed", "model_name": name}

