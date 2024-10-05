import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import numpy as np
from app.commons.commons import pad_vector

# Configurer le logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Fonction de normalisation min-max
def min_max_normalize(data):
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return data
    return (data - min_val) / (max_val - min_val)

# Définition du réseau de neurones avec LSTM
class LSTMNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)  # Couche cachée supplémentaire
        self.fc2 = nn.Linear(128, output_size)  # Couche de sortie

    def forward(self, x):
        x = x.unsqueeze(1)  # Ajouter une dimension pour la séquence (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Garder la dernière sortie temporelle
        out = self.fc1(lstm_out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out

# Fonction de perte hybride avec MSE et Cosine Similarity
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        # Calculer la similarité cosinus en utilisant PyTorch
        cos_sim = nn.functional.cosine_similarity(output, target, dim=0)
        mse_loss = self.mse_loss(output, target)
        # Combinaison des deux pertes : 0.5 * MSE + 0.5 * (1 - Cosine Similarity)
        return 0.5 * mse_loss + 0.5 * (1 - cos_sim.mean())  # Moyenne pour s'assurer d'un scalaire

# Fonction de recherche avec Cosine Similarity et Distance Euclidienne
def search_with_similarity(nn_model, search_vector, indexed_dictionary):
    padded_search_vector = torch.Tensor(search_vector).unsqueeze(0)  # Ajouter la dimension batch_size
    best_match = None
    best_score = float('inf')  # Utiliser une grande valeur pour initialiser

    # Comparer chaque vecteur du dictionnaire
    for vector in indexed_dictionary:
        padded_vector = torch.Tensor(vector).unsqueeze(0)
        with torch.no_grad():
            # Calcul de la similarité cosinus
            cos_similarity = nn.functional.cosine_similarity(nn_model(padded_vector), padded_search_vector).item()
            
            # Calcul de la distance euclidienne
            euclidean_distance = torch.dist(nn_model(padded_vector), padded_search_vector).item()

            # Combinaison des deux : par exemple, 0.5 * Cosine Similarity + 0.5 * (1 / Euclidean Distance)
            score = 0.5 * (1 - cos_similarity) + 0.5 * euclidean_distance

            # Sélectionner le vecteur avec le score le plus bas
            if score < best_score:
                best_score = score
                best_match = vector

    return {
        "best_match": best_match,  # Renvoie uniquement le vecteur d'indices
        "similarity_score": best_score
    }

# Fonction pour entraîner le modèle LSTM avec Early Stopping
def train_lstm_model_nn(train_data, vector_size, epochs=2000, learning_rate=0.001, patience=10, improvement_threshold=0.00001):
    try:
        input_size = vector_size
        hidden_size = 128  # Taille de la couche cachée
        output_size = vector_size  # Taille de sortie égale à celle du vecteur
        model = LSTMNN(input_size, hidden_size, output_size)

        # Critère de perte hybride (MSE et Cosine Similarity)
        criterion = HybridLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Conversion des données en tenseurs
        inputs = torch.Tensor(np.array([min_max_normalize(pad_vector(pair[0], vector_size)) for pair in train_data]))
        targets = torch.Tensor(np.array([min_max_normalize(pad_vector(pair[1], vector_size)) for pair in train_data]))

        losses = []
        start_time = time.time()

        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            losses.append(current_loss)

            if (epoch + 1) % 10 == 0:
                logging.debug(f"Époque {epoch + 1}/{epochs}, Perte: {current_loss}")

            if current_loss < best_loss - improvement_threshold:
                best_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info(f"Arrêt anticipé à l'époque {epoch + 1}. Perte optimale atteinte : {best_loss:.6f}")
                break

    except RuntimeError as e:
        raise RuntimeError(f"Erreur lors de l'entraînement du modèle : {str(e)}") from e

    except Exception as e:
        raise RuntimeError(f"Erreur inattendue lors de l'entraînement : {str(e)}") from e

    total_training_time = time.time() - start_time
    total_parameters = sum(p.numel() for p in model.parameters())

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
