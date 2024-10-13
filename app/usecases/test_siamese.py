from app.commons.commons import create_indexed_glossary
from app.machine_learning.neural_network_siamese import evaluate_similarity
from app.repositories.memory import get_model
from app.services.logger import logger

def test_siamese(name, test_data):
    try:
        total_error = 0
        correct_predictions = 0
        total_tests = len(test_data)
        model = get_model(name)
        glossary = model.get("glossary", [])
        nn_model = model.get("nn_model", None)
        word2idx = create_indexed_glossary(glossary)
        for vector1, vector2, expected_similarity in test_data:
            predicted_similarity = evaluate_similarity(nn_model, vector1, vector2, word2idx)
            error = abs(predicted_similarity - expected_similarity)
            total_error += error
            # Considérer la prédiction correcte si la différence est inférieure à un seuil (par exemple 0.1)
            if error <= 0.1:
                correct_predictions += 1
            # Utiliser le logger pour les sorties
            logger.info(f"Requête: {vector1}, Image: {vector2}")
            logger.info(f"Similarité attendue: {expected_similarity*100}%, Similarité donnée par le modèle: {predicted_similarity*100:.2f}%, Erreur: {error*100:.2f}%")
        avg_error = total_error / total_tests
        # Exprimer avg_error en pourcentage de précision
        precision_percentage = (1 - avg_error) * 100  # Plus avg_error est faible, plus la précision est élevée
        # Afficher le nombre de prédictions correctes sur le nombre total d'essais
        logger.info(f"Nombre de prédictions correctes: {correct_predictions}/{total_tests}")
        logger.info(f"Précision moyenne du modèle sur le jeu de test: {precision_percentage:.2f}%")
    except Exception as e:
        # Gestion des erreurs générales
        logger.error(f"Une erreur s'est produite pendant test_siamese : {str(e)}")
        raise Exception(f"Une erreur s'est produite pendant test_siamese : {str(e)}")
