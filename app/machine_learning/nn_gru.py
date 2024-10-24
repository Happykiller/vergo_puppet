import torch
import torch.nn as nn
import torch.optim as optim
from app.services.logger import logger

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate=0.5):
        """
        Modèle de classification basé sur un GRU avec régularisation par Dropout.
        :param vocab_size: Taille du vocabulaire.
        :param embedding_dim: Dimension des embeddings.
        :param hidden_dim: Dimension des états cachés du GRU.
        :param num_classes: Nombre de classes à prédire.
        :param dropout_rate: Taux de Dropout pour la régularisation.
        """
        super(GRUClassifier, self).__init__()
        # Couche d'embedding pour convertir les indices de mots en vecteurs denses.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Couche GRU pour capturer les dépendances séquentielles.
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # Couche de Dropout pour la régularisation.
        self.dropout = nn.Dropout(dropout_rate)
        # Couche fully connected pour la classification.
        self.fc = nn.Linear(hidden_dim, num_classes)
        # Fonction d'activation LogSoftmax pour obtenir des probabilités logarithmiques.
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        """
        Passage avant du modèle.
        :param x: Séquences d'indices de mots (batch_size x seq_length).
        :return: Log-probabilités pour chaque classe (batch_size x num_classes).
        """
        # Conversion des indices de mots en embeddings.
        embedded = self.embedding(x)  # Taille : (batch_size, seq_length, embedding_dim)
        # Passage à travers la couche GRU.
        _, hidden = self.gru(embedded)  # hidden taille : (1, batch_size, hidden_dim)
        # Appliquer le Dropout à l'état caché.
        hidden = self.dropout(hidden.squeeze(0))  # Taille : (batch_size, hidden_dim)
        # Passage à travers la couche fully connected.
        output = self.fc(hidden)  # Taille : (batch_size, num_classes)
        # Application de la fonction d'activation.
        output = self.log_softmax(output)
        return output

def train_gru(model, sequences, labels, num_epochs, batch_size, learning_rate):
    """
    Entraîne le modèle GRU avec régularisation et hyperparamètres ajustés.
    :param model: Modèle GRU à entraîner.
    :param sequences: Tenseur des séquences d'entraînement.
    :param labels: Tenseur des labels correspondants.
    :param num_epochs: Nombre d'époques pour l'entraînement.
    :param batch_size: Taille des lots.
    :param learning_rate: Taux d'apprentissage.
    """
    criterion = nn.NLLLoss()  # Fonction de perte appropriée pour la classification multi-classe
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Création du DataLoader pour gérer les lots
    dataset = torch.utils.data.TensorDataset(sequences, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Boucle d'entraînement avec suivi des pertes pour l'arrêt anticipé
    best_loss = float('inf')
    patience = 5  # Nombre d'époques sans amélioration après lequel arrêter l'entraînement
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()  # Mettre le modèle en mode entraînement
        total_loss = 0
        for batch_sequences, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Époque {epoch+1}/{num_epochs}, Perte moyenne: {avg_loss:.4f}")
        
        # Vérification de l'amélioration
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Arrêt anticipé si aucune amélioration
        if epochs_without_improvement >= patience:
            logger.info("Arrêt anticipé de l'entraînement en raison d'aucune amélioration de la perte.")
            break