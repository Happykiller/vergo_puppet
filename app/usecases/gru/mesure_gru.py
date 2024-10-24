import torch
from typing import List
from app.services.logger import logger
from app.repositories.memory import get_model
from app.apis.models.gru_training_data import GRUTrainingData
from app.machine_learning.nn_gru import GRUClassifier
from sklearn.metrics import classification_report

def mesure_gru(name: str, test_data: List[GRUTrainingData]):
    """
    Mesure les performances du modèle GRU sur les données de test fournies.
    :param name: Nom du modèle.
    :param test_data: Liste des données de test.
    """
    try:
        model_data = get_model(name)
        nn_model_state = model_data.get("nn_model", None)
        if nn_model_state is None:
            raise Exception("Modèle non entraîné")
        
        word2idx = model_data.get("word2idx", None)
        idx2category = model_data.get("idx2category", None)
        category2idx = model_data.get("category2idx", None)
        if word2idx is None or idx2category is None or category2idx is None:
            raise Exception("Données du modèle incomplètes")
        
        # Charger le modèle avec les paramètres sauvegardés
        vocab_size = len(word2idx)
        num_classes = len(idx2category)
        embedding_dim = 128
        hidden_dim = 256
        dropout_rate = 0.5

        # Récupérer les hyperparamètres sauvegardés
        hyperparameters = model_data.get("hyperparameters", None)
        if hyperparameters is None:
            raise Exception("Hyperparamètres non trouvés dans les données du modèle.")

        embedding_dim = hyperparameters['embedding_dim']
        hidden_dim = hyperparameters['hidden_dim']
        dropout_rate = hyperparameters['dropout_rate']

        # Charger le modèle avec les hyperparamètres appropriés
        model_gru = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate)
        model_gru.load_state_dict(nn_model_state)
        model_gru.eval()
        
        total_error = 0
        correct_predictions = 0
        total_tests = len(test_data)
        
        y_true = []
        y_pred = []
        
        for data in test_data:
            # Préparer les données d'entrée
            tokens = data.tokens
            expected_category = data.category
            input_tensor = process_input(tokens, word2idx)
            
            # Prédiction
            with torch.no_grad():
                outputs = model_gru(input_tensor)
                predicted_idx = torch.argmax(outputs, dim=1).item()
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
        
        # Calculer des métriques détaillées
        report = classification_report(y_true, y_pred, target_names=idx2category.values())
        logger.info("Rapport de classification détaillé :")
        logger.info(f"\n{report}")
    except Exception as e:
        logger.error(f"Une erreur s'est produite pendant la mesure : {str(e)}")
        raise Exception(f"Une erreur s'est produite pendant la mesure : {str(e)}")

def process_input(tokens: List[str], word2idx):
    """
    Transforme la liste de tokens en tenseur d'indices avec padding.
    :param tokens: Liste de tokens de la séquence à classer.
    :param word2idx: Dictionnaire de mapping mot->indice.
    :return: Tenseur de la séquence préparée.
    """
    seq = [word2idx.get(token, word2idx['<PAD>']) for token in tokens]
    max_seq_length = len(seq)
    seq += [word2idx['<PAD>']] * (max_seq_length - len(seq))  # Padding si nécessaire
    seq_tensor = torch.tensor([seq], dtype=torch.long)
    return seq_tensor
