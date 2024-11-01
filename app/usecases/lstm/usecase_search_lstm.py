import joblib
import numpy as np
import pandas as pd
from app.repositories.memory import get_model
from fastapi import HTTPException # type: ignore
from app.machine_learning.nn_lstm import predict_nn_lstm
from app.apis.models.weather_model_data import WeatherSearchModelData
from app.usecases.lstm.usecase_commons_lstm import inverse_transform_predictions, preprocess_input_data

def search_lstm(name: str, input_data: WeatherSearchModelData):
    """
    Utilise le modèle LSTM pour prédire la température à partir des données fournies.
    :param name: Nom du modèle
    :param input_data: Données d'entrée sans la température
    :return: Prédiction de la température
    """
    # Charger le modèle
    model_data = get_model(name)
    if model_data is None or not model_data:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    nn_model = model_data['nn_model']
    nn_model.eval()

    # Charger le scaler et l'encodeur
    scaler = joblib.load(f'{name}_scaler.pkl')
    target_scaler = joblib.load(f'{name}_target_scaler.pkl')
    coco_encoder = joblib.load(f'{name}_coco_encoder.pkl')

    # Préparer les données d'entrée
    df_input = pd.DataFrame([input_data.dict()])
    df_processed = preprocess_input_data(df_input, scaler, coco_encoder)

    # Vérifier la longueur de séquence attendue par le modèle
    sequence_length = 24  # À ajuster si nécessaire

    # Créer une séquence en dupliquant l'entrée pour atteindre la longueur nécessaire
    input_sequence = np.repeat(df_processed.values, sequence_length, axis=0)
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Shape: (1, sequence_length, num_features)

    # Faire la prédiction
    prediction_normalized = predict_nn_lstm(nn_model, input_sequence)

    # Inverser la normalisation de la prédiction
    prediction_inverse = inverse_transform_predictions(prediction_normalized, target_scaler)[0]

    # Convertir en float pour la sérialisation JSON
    prediction_value = float(prediction_inverse)

    return {"prediction": prediction_value}
