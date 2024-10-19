import joblib
from app.machine_learning.neural_network_simple import predict
from app.repositories.memory import get_model
from app.services.logger import logger
from app.usecases.simple_nn.commons import process_input_data

def mesure_simple_nn(name, test_data):
    try:
        total_error = 0
        total_percentage_error = 0
        correct_predictions = 0
        total_tests = len(test_data)
        model = get_model(name)
        nn_model = model.get("nn_model", None)
        if nn_model is None:
            raise Exception("Model not trained yet")
        
        # Charger l'encodeur, le scaler et les indices
        encoder_filename = model.get("encoder_filename")
        scaler_filename = model.get("scaler_filename")
        indices_filename = model.get("indices_filename")
        if not encoder_filename or not scaler_filename or not indices_filename:
            raise Exception("Missing encoder, scaler, or indices in the model")
        
        encoder = joblib.load(encoder_filename)
        scaler = joblib.load(scaler_filename)
        indices_info = joblib.load(indices_filename)
        categorical_indices = indices_info["categorical_indices"]
        numerical_indices = indices_info["numerical_indices"]
        
        # Récupérer les paramètres de normalisation des targets
        targets_mean = model.get("targets_mean")
        targets_std = model.get("targets_std")
        if targets_mean is None or targets_std is None:
            raise Exception("Missing normalization parameters in the model")
        
        for data in test_data:
            # Préparer les données d'entrée
            input_data = [
                data.type,
                data.surface,
                data.pieces,
                data.floor,
                data.parking,
                data.balcon,
                data.ascenseur,
                data.orientation,
                data.transports,
                data.neighborhood
            ]
            expected = data.price
            
            # Transformer les données d'entrée
            input_processed = process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices)
            
            # Prédiction
            predicted = predict(nn_model, input_processed, targets_mean, targets_std)
            
            # Calcul de l'erreur
            error = abs(predicted - expected)
            percentage_error = (error / expected) * 100
            total_error += error
            total_percentage_error += percentage_error
            
            # Considérer la prédiction correcte si l'erreur est inférieure ou égale à 10% du prix réel
            if percentage_error <= 10:
                correct_predictions += 1
            
            # Utiliser le logger pour les sorties
            logger.info(f"Requête: {input_data}")
            logger.info(f"Prix attendu: {expected}€, Prix donné par le modèle: {predicted:.2f}€, Erreur: {error:.2f}€, Erreur en %: {percentage_error:.2f}%")
        
        avg_error = total_error / total_tests
        avg_percentage_error = total_percentage_error / total_tests
        # Afficher le nombre de prédictions correctes sur le nombre total d'essais
        logger.info(f"Nombre de prédictions correctes: {correct_predictions}/{total_tests}")
        logger.info(f"Erreur absolue moyenne (MAE) sur le jeu de test: {avg_error:.2f}€")
        logger.info(f"Erreur absolue moyenne en pourcentage (MAPE) sur le jeu de test: {avg_percentage_error:.2f}%")
    except Exception as e:
        logger.error(f"Une erreur s'est produite pendant la mesure : {str(e)}")
        raise Exception(f"Une erreur s'est produite pendant la mesure : {str(e)}")
