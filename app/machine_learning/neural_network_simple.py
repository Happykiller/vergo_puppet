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

# Fonction pour entraîner le modèle SimpleNN avec Early Stopping
def train_model_nn(features_processed, targets_standardized, input_size, epochs=3000, learning_rate=0.001, patience=10, improvement_threshold=0.00001):
    """
    Entraîne le modèle de réseau de neurones avec les features et les targets fournis.
    """
    try:
        # Initialisation des tailles
        hidden_size = 128  # Taille des couches cachées
        output_size = 1  # La taille de sortie doit être 1 (pour prédire un prix unique)

        # Création du modèle SimpleNN
        model = SimpleNN(input_size, hidden_size, output_size)

        # Utilisation de la fonction de perte
        criterion = nn.MSELoss()
        # Optimiseur Adam avec un taux d'apprentissage initial
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Conversion en tenseurs
        inputs_tensor = torch.tensor(features_processed, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_standardized, dtype=torch.float32).unsqueeze(1)

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

    except RuntimeError as e:
        raise RuntimeError(f"Erreur lors de l'entraînement du modèle : {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Erreur inattendue lors de l'entraînement : {str(e)}") from e

# Fonction pour faire une prédiction avec le modèle SimpleNN
def predict(model, input_processed, targets_mean, targets_std):
    try:
        # Conversion en tenseur
        input_tensor = torch.tensor(input_processed, dtype=torch.float32)
        
        # Prédiction
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Conversion du tenseur de sortie en valeur scalaire
        predicted_standardized = output_tensor.item()
        
        # Déstandardisation de la prédiction
        predicted_price = predicted_standardized * targets_std + targets_mean
        
        return float(predicted_price)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise
