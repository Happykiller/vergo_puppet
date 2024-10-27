import joblib
import numpy as np
from app.services.logger import logger
from app.repositories.memory import get_model
from app.machine_learning.nn_lstm import predict_nn_lstm
from app.apis.models.weather_data_model import WeatherData
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from app.usecases.lstm.usecase_commons_lstm import inverse_transform_predictions, preprocess_input_data
from typing import List
import pandas as pd


def mesure_lstm(name: str, test_data: List[WeatherData]):
    try:
        # Vérifier que le modèle existe
        model = get_model(name)
        nn_model = model.get("nn_model", None)
        if nn_model is None:
            raise Exception("Le modèle n'a pas encore été entraîné")
        
        # Charger le scaler et l'encodeur
        scaler = joblib.load(f'{name}_scaler.pkl')
        target_scaler = joblib.load(f'{name}_target_scaler.pkl')
        coco_encoder = joblib.load(f'{name}_coco_encoder.pkl')
        
        # Définir la longueur de séquence utilisée lors de l'entraînement
        sequence_length = 24  # Ajuster si nécessaire

        # Listes pour stocker les valeurs réelles et prédites
        y_true_list = []
        y_pred_list = []
        
        # Boucler sur chaque échantillon de données de test
        for data in test_data:
            # Extraire la valeur réelle de 'temp'
            y_true = data.temp
            
            # Préparer les données d'entrée (sans 'temp')
            data_dict = data.dict()
            data_dict.pop('temp', None)  # Supprimer 'temp' des données d'entrée
            df_input = pd.DataFrame([data_dict])
            
            # Prétraiter les données d'entrée
            df_processed = preprocess_input_data(df_input, scaler, coco_encoder)
            
            # Créer une séquence en dupliquant l'entrée pour atteindre la longueur nécessaire
            input_sequence = np.repeat(df_processed.values, sequence_length, axis=0)
            input_sequence = np.expand_dims(input_sequence, axis=0)  # Shape: (1, sequence_length, num_features)
            
            # Faire la prédiction
            prediction_normalized = predict_nn_lstm(nn_model, input_sequence)
            
            # Inverser la normalisation de la prédiction
            prediction_inverse = inverse_transform_predictions(prediction_normalized, target_scaler)[0]
            
            # Ajouter les valeurs à la liste
            y_true_list.append(y_true)
            y_pred_list.append(prediction_inverse)
        
        # Convertir les listes en tableaux numpy
        y_true_array = np.array(y_true_list)
        y_pred_array = np.array(y_pred_list)
        
        # Calculer les métriques de performance
        mae = mean_absolute_error(y_true_array, y_pred_array)
        mape = mean_absolute_percentage_error(y_true_array, y_pred_array) * 100  # En pourcentage
        
        # Afficher les résultats
        logger.info(f"Erreur absolue moyenne (MAE) sur le jeu de test: {mae:.2f}")
        logger.info(f"Erreur absolue moyenne en pourcentage (MAPE) sur le jeu de test: {mape:.2f}%")
        
        # Retourner les métriques
        return {
            "mae": mae,
            "mape": mape,
            "nombre_de_tests": len(y_pred_list)
        }
    except Exception as e:
        logger.error(f"Une erreur s'est produite pendant la mesure : {str(e)}")
        raise Exception(f"Une erreur s'est produite pendant la mesure : {str(e)}")

