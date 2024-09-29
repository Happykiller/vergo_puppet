import torch
import torch.nn as nn
import torch.optim as optim
import time  # Importer pour le suivi du temps
import logging  # Importer la bibliothèque logging
import numpy as np  # Importer numpy pour la normalisation

from app.commons.commons import pad_vector

# Configurer le logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Fonction de normalisation min-max
def min_max_normalize(data):
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return data  # Éviter la division par 0 si les données ont la même valeur
    return (data - min_val) / (max_val - min_val)

# Définition du réseau de neurones
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Fonction pour entraîner le modèle avec early stopping
def train_model_nn(train_data, vector_size, epochs=1000, learning_rate=0.001, patience=10, improvement_threshold=0.00001):
    try:
        input_size = vector_size  # Taille maximale du vecteur d'entrée
        hidden_size = 128  # Taille de la couche cachée
        output_size = vector_size  # Taille de sortie égale à celle du vecteur
        model = SimpleNN(input_size, hidden_size, output_size)

        # Critère de perte et optimiseur
        criterion = nn.MSELoss()  # Erreur quadratique moyenne pour minimiser la différence entre vecteurs
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Convertir les données en numpy.ndarray puis en tenseurs PyTorch
        inputs_np = np.array([min_max_normalize(pad_vector(pair[0], vector_size)) for pair in train_data])
        targets_np = np.array([min_max_normalize(pad_vector(pair[1], vector_size)) for pair in train_data])

        inputs = torch.Tensor(inputs_np)
        targets = torch.Tensor(targets_np)

        # Début du suivi du temps
        start_time = time.time()

    except IndexError as e:
        raise ValueError(f"Erreur dans la méthode 'train_model_nn': Problème avec les indices dans les données d'entraînement. Détails : {str(e)}") from e

    except TypeError as e:
        raise ValueError(f"Erreur dans la méthode 'train_model_nn': Type de données incorrect. Vérifiez que 'train_data' est une liste de tuples (input, target). Détails : {str(e)}") from e

    except Exception as e:
        raise RuntimeError(f"Erreur inattendue dans la méthode 'train_model_nn': {str(e)}") from e

    # Entraînement du modèle avec gestion du early stopping
    try:
        losses = []  # Stocker la perte à chaque époque
        best_loss = float('inf')  # Initialiser avec une perte très élevée
        epochs_without_improvement = 0  # Compteur pour le nombre d'epochs sans amélioration

        for epoch in range(epochs):
            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs = model(inputs)  # Propagation avant
            loss = criterion(outputs, targets)  # Calcul de la perte
            loss.backward()  # Rétropropagation pour calculer les gradients
            optimizer.step()  # Mise à jour des poids

            current_loss = loss.item()
            losses.append(current_loss)

            # Afficher la perte toutes les 10 époques pour le suivi
            if (epoch + 1) % 10 == 0:
                logging.debug(f"Époque {epoch + 1}/{epochs}, Perte: {current_loss}")

            # Early Stopping : vérifier si la perte s'améliore
            if current_loss < best_loss - improvement_threshold:
                best_loss = current_loss
                epochs_without_improvement = 0  # Réinitialiser le compteur
            else:
                epochs_without_improvement += 1

            # Vérifier si on doit arrêter l'entraînement
            if epochs_without_improvement >= patience:
                logging.info(f"Arrêt anticipé à l'époque {epoch + 1}. Perte optimale atteinte : {best_loss:.6f}")
                break

    except RuntimeError as e:
        raise RuntimeError(f"Erreur lors de l'entraînement du modèle : {str(e)}") from e

    except Exception as e:
        raise RuntimeError(f"Erreur inattendue lors de l'entraînement : {str(e)}") from e

    # Calcul du temps total d'entraînement
    total_training_time = time.time() - start_time

    # Calcul du nombre total de paramètres (poids)
    total_parameters = sum(p.numel() for p in model.parameters())

    # Statistiques sur la perte
    avg_loss = sum(losses) / len(losses)
    min_loss = min(losses)
    max_loss = max(losses)

    logging.info(f"Temps total d'entraînement: {total_training_time:.2f} secondes")
    logging.info(f"Nombre total de paramètres: {total_parameters}")
    logging.info(f"Perte moyenne: {avg_loss}")
    logging.info(f"Perte minimale: {min_loss}")
    logging.info(f"Perte maximale: {max_loss}")
    logging.info(f"Perte finale après {len(losses)} epochs : {losses[-1]}")

    return model, losses

# Fonction pour faire une prédiction
def predict(model, input_vector):
    input_tensor = torch.Tensor(input_vector)
    with torch.no_grad():
        output_vector = model(input_tensor)
    
    logging.debug(f"Vecteur de sortie prédit : {output_vector}")
    return output_vector
