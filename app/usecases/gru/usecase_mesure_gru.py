from typing import List
from app.services.logger import logger
from app.repositories.memory import get_model
from app.machine_learning.nn_gru import predict
from app.usecases.gru.usecase_commons_gru import process_input
from app.apis.models.gru_training_model_data import GRUTrainingModelData

def mesure_gru(name: str, test_data: List[GRUTrainingModelData]):
    """
    Mesure les performances du modèle GRU sur les données de test fournies.
    :param name: Nom du modèle.
    :param test_data: Liste des données de test.
    """
    try:
        model_data = get_model(name)
        nn_model = model_data.get("nn_model", None)
        if nn_model is None:
            raise Exception("Modèle non entraîné")
        
        word2idx = model_data.get("word2idx", None)
        idx2category = model_data.get("idx2category", None)
        category2idx = model_data.get("category2idx", None)
        if word2idx is None or idx2category is None or category2idx is None:
            raise Exception("Données du modèle incomplètes")
        
        total_error = 0
        correct_predictions = 0
        total_tests = len(test_data)
        
        y_true = []
        y_pred = []
        
        for data in test_data:
            # Préparer les données d'entrée
            tokens = data.tokens
            expected_category = data.category
            input = process_input(tokens, word2idx)
            
            # Prédiction
            predicted_idx = predict(nn_model, input)
            predicted_category = idx2category[predicted_idx]
            
            # Après
            try:
                y_true.append(category2idx[expected_category])
            except KeyError:
                logger.warning(f"Catégorie inconnue dans les données de test : '{expected_category}'. Elle n'a pas été vue pendant l'entraînement.")
                continue  # Ignorer cet échantillon
            y_pred.append(predicted_idx)
            
            # Vérification de la prédiction
            if predicted_category == expected_category:
                correct_predictions += 1
            else:
                total_error += 1
            
            # Log des résultats individuels
            logger.info(f"Requête: {tokens}")
            logger.info(f"Catégorie attendue: {expected_category}, Catégorie prédite: {predicted_category}")
        
        # Résumé des performances
        logger.info(f"Nombre de prédictions correctes: {correct_predictions}/{total_tests}")
        accuracy = correct_predictions / total_tests * 100
        logger.info(f"Taux de précision du modèle : {accuracy:.2f}%")
    except Exception as e:
        logger.error(f"Une erreur s'est produite pendant la mesure : {str(e)}")
        raise Exception(f"Une erreur s'est produite pendant la mesure : {str(e)}")
