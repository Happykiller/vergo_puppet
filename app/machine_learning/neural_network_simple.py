import torch
import torch.nn as nn
import torch.optim as optim
import time
from app.services.logger import logger
import numpy as np

# Fonction pour normaliser les données entre 0 et 1 (Min-Max scaling)
def min_max_normalize(data):
    data = np.array(data, dtype=np.float32)
    min_val = np.min(data, axis=0)  # Minimum pour chaque caractéristique
    max_val = np.max(data, axis=0)  # Maximum pour chaque caractéristique
    # Éviter la division par zéro si min == max
    return (data - min_val) / (max_val - min_val + 1e-8)  # Ajout d'un petit terme pour éviter la division par 0

# Définition du réseau de neurones SimpleNN amélioré avec plusieurs couches et du Dropout pour la régularisation
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialisation du réseau SimpleNN avec :
        - input_size : taille du vecteur d'entrée
        - hidden_size : nombre de neurones dans la couche cachée
        - output_size : taille du vecteur de sortie
        """
        super(SimpleNN, self).__init__()
        # Première couche fully connected
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Fonction d'activation ReLU après la première couche
        self.relu1 = nn.ReLU()
        # Deuxième couche fully connected pour ajouter de la profondeur au réseau
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Fonction d'activation ReLU après la deuxième couche
        self.relu2 = nn.ReLU()
        # Troisième couche fully connected (couche de sortie)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # Dropout pour réduire le surapprentissage (50% des neurones désactivés aléatoirement)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Fonction de passage avant (forward) :
        - Prend un vecteur d'entrée x et le passe à travers les couches du réseau.
        - Applique des fonctions d'activation ReLU et un Dropout pour la régularisation.
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)  # Applique Dropout après la première couche cachée
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)  # Pas de fonction d'activation après la dernière couche
        return out

# Définition de la fonction de perte hybride combinant MSE (Mean Squared Error) et Cosine Similarity
class HybridLoss(nn.Module):
    def __init__(self):
        """
        Initialisation de la fonction de perte hybride :
        - Combine la perte MSE (pour la différence en magnitude) et
        - la similarité cosinus (pour la différence en direction).
        """
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        """
        Calcul de la perte hybride :
        - Calcule d'abord la similarité cosinus entre output et target.
        - Calcule ensuite la perte MSE.
        - Combine les deux avec un poids de 0.5 pour chacun.
        """
        # Calcul de la similarité cosinus entre output et target
        cos_sim = nn.functional.cosine_similarity(output, target, dim=0)
        # Calcul de la perte MSE
        mse_loss = self.mse_loss(output, target)
        # Combinaison : 50% MSE et 50% (1 - similarité cosinus)
        return 0.5 * mse_loss + 0.5 * (1 - cos_sim.mean())  # Moyenne pour obtenir un scalaire

# Fonction pour entraîner le modèle SimpleNN avec Early Stopping
def train_model_nn(train_data, vector_size, epochs=3000, learning_rate=0.001, patience=10, improvement_threshold=0.00001):
    """
    Fonction d'entraînement du modèle SimpleNN :
    - Prend les données d'entraînement, la taille du vecteur, le nombre d'époques et le taux d'apprentissage.
    - Utilise une stratégie d'arrêt anticipé (Early Stopping) pour arrêter l'entraînement si la perte n'améliore plus.
    - Retourne le modèle entraîné et les pertes sur chaque époque.
    """
    try:
        # Initialisation des tailles
        input_size = vector_size  # Taille des vecteurs d'entrée
        hidden_size = 128  # Taille des couches cachées
        output_size = 1  # La taille de sortie doit être 1 (pour prédire un prix unique)

        # Création du modèle SimpleNN
        model = SimpleNN(input_size, hidden_size, output_size)

        # Utilisation de la fonction de perte hybride
        criterion = HybridLoss()
        # Optimiseur Adam avec un taux d'apprentissage initial
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        logger.debug(f"train_data {train_data}")

        # Séparation des features (inputs) et des targets (prix)
        features = np.array([x[0] for x in train_data], dtype=np.float32)  # Entrées
        targets = np.array([x[1] for x in train_data], dtype=np.float32)  # Cibles (prix)

        # Normalisation des features (input data)
        features_normalized = min_max_normalize(features)

        # Conversion en tenseurs
        inputs_tensor = torch.tensor(features_normalized, dtype=torch.float32)  # Les entrées sous forme de tenseurs
        targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # Cibles, avec une dimension ajoutée

        # Liste pour stocker la perte à chaque époque
        losses = []
        start_time = time.time()

        best_loss = float('inf')  # Initialisation de la meilleure perte à une grande valeur
        epochs_without_improvement = 0  # Compteur pour l'arrêt anticipé

        # Boucle d'entraînement
        for epoch in range(epochs):
            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs = model(inputs_tensor)  # Passer les entrées dans le modèle
            loss = criterion(outputs, targets_tensor)  # Calculer la perte
            loss.backward()  # Calculer les gradients
            optimizer.step()  # Mettre à jour les poids

            current_loss = loss.item()  # Récupérer la perte actuelle
            losses.append(current_loss)

            # Afficher la perte toutes les 10 époques
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Époque {epoch + 1}/{epochs}, Perte: {current_loss}")

            # Vérification de l'amélioration de la perte
            if current_loss < best_loss - improvement_threshold:
                best_loss = current_loss
                epochs_without_improvement = 0  # Réinitialiser si la perte s'améliore
            else:
                epochs_without_improvement += 1  # Incrémenter si aucune amélioration

            # Arrêter l'entraînement si la perte n'améliore plus
            if epochs_without_improvement >= patience:
                logger.info(f"Arrêt anticipé à l'époque {epoch + 1}. Perte optimale atteinte : {best_loss:.6f}")
                break

    except RuntimeError as e:
        raise RuntimeError(f"Erreur lors de l'entraînement du modèle : {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Erreur inattendue lors de l'entraînement : {str(e)}") from e

    # Calcul du temps total d'entraînement
    total_training_time = time.time() - start_time
    total_parameters = sum(p.numel() for p in model.parameters())

    # Statistiques sur les pertes
    avg_loss = sum(losses) / len(losses)
    min_loss = min(losses)
    max_loss = max(losses)

    # Log des statistiques d'entraînement
    logger.info(f"Temps total d'entraînement: {total_training_time:.2f} secondes")
    logger.info(f"Nombre total de paramètres: {total_parameters}")
    logger.info(f"Perte moyenne: {avg_loss}")
    logger.info(f"Perte minimale: {min_loss}")
    logger.info(f"Perte maximale: {max_loss}")
    logger.info(f"Perte finale après {len(losses)} epochs : {losses[-1]}")

    return model, losses

# Fonction pour faire une prédiction avec le modèle SimpleNN
def predict(model, input_vector):
    """
    Fonction de prédiction avec normalisation des entrées.
    """
    try:
        # Conversion de l'input en un numpy array et normalisation
        input_array = np.array(input_vector, dtype=np.float32).reshape(1, -1)  # Ajouter une dimension batch
        input_array_normalized = min_max_normalize(input_array)  # Normalisation des données

        # Passer les données normalisées à travers le modèle
        with torch.no_grad():  # Désactiver le calcul des gradients pour la prédiction
            output_tensor = model(torch.tensor(input_array_normalized))

        # Conversion du tenseur de sortie en une valeur Python standard (target_price)
        predicted_price = output_tensor.detach().cpu().numpy().squeeze()

        # Vérification que la sortie est bien un scalaire
        if predicted_price.size == 1:
            return float(predicted_price)  # Retourne la valeur prédite comme un float
        else:
            raise ValueError("La sortie du modèle n'est pas une valeur scalaire unique")
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise