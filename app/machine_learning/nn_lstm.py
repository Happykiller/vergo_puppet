import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.services.logger import logger

class LSTMNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

def train_nn_lstm(X_train, y_train, epochs=20, learning_rate=0.001, patience=10):
    start_time = time.time()

    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    
    # Définition des paramètres du modèle
    input_size = X_train.shape[2]  # Nombre de caractéristiques
    hidden_size = 64  # À ajuster selon les besoins
    
    nn_model = LSTMNN(input_size, hidden_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)

    # Variables pour l'arrêt anticipé
    best_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    # Entraînement du modèle
    for epoch in range(epochs):
        nn_model.train()
        optimizer.zero_grad()
        outputs = nn_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Vérification de l'amélioration
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch + 1
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            best_model_state = nn_model.state_dict()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Époque {epoch+1}/{epochs}, Perte: {loss.item():.6f}")

        # Vérification de la patience
        if patience_counter >= patience:
            logger.info(f"Aucune amélioration pendant {patience} époques consécutives. Arrêt anticipé à l'époque {epoch+1}.")
            break

    # Charger le meilleur modèle
    if best_model_state is not None:
        nn_model.load_state_dict(best_model_state)
        logger.info(f"Meilleur modèle obtenu à l'époque {best_epoch} avec une perte de validation de {best_loss:.6f}")
    else:
        logger.warning("Aucune amélioration observée pendant l'entraînement.")

    # Calcul du temps total d'entraînement
    total_training_time = time.time() - start_time

    # Log des statistiques d'entraînement
    logger.info(f"Temps total d'entraînement: {total_training_time:.2f} secondes")
    
    return nn_model

def predict_nn_lstm(nn_model, X_input):
    """
    Utilise le modèle LSTM entraîné pour faire des prédictions sur de nouvelles données.
    :param nn_model: Modèle LSTM entraîné
    :param X_input: Données d'entrée, de forme (batch_size, sequence_length, input_size)
    :return: Prédictions sous forme de tableau numpy
    """
    nn_model.eval()  # Mettre le modèle en mode évaluation

    # Conversion en tenseur PyTorch
    X_input_tensor = torch.Tensor(X_input)

    # Désactiver le calcul des gradients pour gagner du temps et de la mémoire
    with torch.no_grad():
        predictions = nn_model(X_input_tensor)

    # Conversion des prédictions en numpy array
    predictions_np = predictions.numpy().reshape(-1)

    return predictions_np