import joblib
import numpy as np
import pandas as pd
from typing import List
from app.apis.models.weather_data import WeatherData
from app.machine_learning.nn_lstm import train_nn_lstm
from app.services.logger import logger
from app.usecases.lstm.usecase_commons_lstm import prepare_sequences, preprocess_data
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import get_model, update_model

def train_lstm(name: str, training_data: List[WeatherData]):
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
    
    logger.info(f"Type de machine learning utilisé pour l'entraînement LSTM")

    # Convertir les données d'entraînement en DataFrame
    df = pd.DataFrame([data.dict() for data in training_data])
    
    # Traitement des données
    df_processed, y_temp_scaled, scaler, target_scaler, coco_encoder = preprocess_data(df)
    
    # Préparation des données pour l'entraînement
    X_train, y_train = prepare_sequences(df_processed, y_temp_scaled)

    # Sauvegarde des objets de prétraitement
    joblib.dump(scaler, f'{name}_scaler.pkl')
    joblib.dump(target_scaler, f'{name}_target_scaler.pkl')
    joblib.dump(coco_encoder, f'{name}_coco_encoder.pkl')

    # Vérifier les NaN dans X_train et y_train
    if np.isnan(X_train).any():
        logger.error("X_train contient des NaN. Entraînement annulé.")
        raise ValueError("X_train contient des NaN.")
    if np.isnan(y_train).any():
        logger.error("y_train contient des NaN. Entraînement annulé.")
        raise ValueError("y_train contient des NaN.")

    # Entraînement du modèle
    nn_model = train_nn_lstm(X_train, y_train)
    
    # Mise à jour du modèle dans le stockage
    update_model(name, {"nn_model": nn_model})
    
    return {"status": "training completed", "model_name": name}