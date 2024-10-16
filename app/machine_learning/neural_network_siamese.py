# neural_network_siamese.py

from app.usecases.tokens_to_indices import tokens_to_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import time  # Pour mesurer le temps d'entraînement
from app.services.logger import logger  # Importation du logger personnalisé

# Définition du modèle Siamese LSTM
class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super(SiameseLSTM, self).__init__()
        # Couche d'embedding pour convertir les indices de mots en vecteurs denses
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM pour encoder les séquences
        self.lstm = nn.LSTM(
            embedding_dim,      # Dimension des embeddings en entrée
            hidden_dim,         # Dimension cachée du LSTM
            batch_first=True,   # Les batches sont en premier dans les dimensions
            bidirectional=False # LSTM unidirectionnel
        )

    def forward_once(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [batch_size, seq_length]
        # Embedding de la séquence d'entrée
        embedded = self.embedding(x)
        # Pack la séquence pour gérer les longueurs variables
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),      # Longueurs des séquences
            batch_first=True,
            enforce_sorted=False
        )
        # Passage à travers le LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # On utilise le dernier état caché comme représentation de la séquence
        output = hidden[-1]   # [batch_size, hidden_dim]
        return output

    def forward(self, input1: torch.Tensor, lengths1: torch.Tensor,
                input2: torch.Tensor, lengths2: torch.Tensor):
        # Encode la première séquence
        output1 = self.forward_once(input1, lengths1)
        # Encode la seconde séquence
        output2 = self.forward_once(input2, lengths2)
        return output1, output2

# Définition de la fonction de perte pour le modèle Siamese
class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        # On utilise la MSELoss pour comparer la similarité prédite et la similarité réelle
        self.mse_loss = nn.MSELoss()

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor):
        # Calcul de la similarité cosinus entre les représentations des deux séquences
        cosine_similarity = F.cosine_similarity(output1, output2)
        # Ajustement de la similarité pour qu'elle soit entre 0 et 1 (au lieu de -1 à 1)
        similarity = (cosine_similarity + 1) / 2
        # Calcul de la perte en comparant la similarité prédite à la similarité réelle
        loss = self.mse_loss(similarity, label)
        return loss

# Dataset personnalisé pour les paires de séquences et leur similarité
class SimilarityDataset(Dataset):
    def __init__(self, data: List[Tuple[List[int], List[int], float]]):
        self.pairs = []
        self.labels = []
        for idxs1, idxs2, label in data:
            # Prépare les séquences et les ajoute à la liste des paires
            torch1 = torch.tensor(idxs1, dtype=torch.long)
            torch2 = torch.tensor(idxs2, dtype=torch.long)
            self.pairs.append(
                (torch1, torch2)
            )
            # Ajoute le label correspondant
            self.labels.append(label)

    def __len__(self):
        # Retourne la taille du dataset
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Retourne la paire de séquences et le label à l'indice donné
        seq1, seq2 = self.pairs[idx]
        label = self.labels[idx]
        return seq1, seq2, label

# Fonction collate_fn pour le DataLoader afin de gérer les séquences de longueur variable
def collate_fn(data):
    # Sépare les données en séquences et labels
    seq1_list, seq2_list, label_list = zip(*data)
    # Traitement des séquences 1
    lengths1 = [len(seq) for seq in seq1_list]
    seq1_padded = nn.utils.rnn.pad_sequence(
        seq1_list, batch_first=True, padding_value=0
    )
    # Traitement des séquences 2
    lengths2 = [len(seq) for seq in seq2_list]
    seq2_padded = nn.utils.rnn.pad_sequence(
        seq2_list, batch_first=True, padding_value=0
    )
    # Conversion des labels en tenseur
    labels = torch.tensor(label_list, dtype=torch.float)
    return (
        seq1_padded,
        torch.tensor(lengths1),
        seq2_padded,
        torch.tensor(lengths2),
        labels,
    )

# Fonction pour entraîner le modèle Siamese LSTM avec early stopping et logging
def train_siamese_model_nn(
    training_data: List[Tuple[List[int], List[int], float]],
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_epochs: int = 200,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    patience: int = 10  # Nombre d'époques sans amélioration avant d'arrêter
):
    # Création du dataset personnalisé
    dataset = SimilarityDataset(training_data)
    # Création du DataLoader pour gérer les batches
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Initialisation du modèle Siamese LSTM
    model = SiameseLSTM(vocab_size, embedding_dim, hidden_dim)
    # Définition de la fonction de perte
    criterion = SimilarityLoss()
    # Définition de l'optimiseur (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Vérification si un GPU est disponible, sinon utilisation du CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Déplacement du modèle sur le device sélectionné
    model = model.to(device)

    # Variables pour l'early stopping
    best_loss = float('inf')   # Initialisation de la meilleure perte à l'infini
    patience_counter = 0       # Compteur d'époques sans amélioration
    losses = []                # Liste pour stocker la perte moyenne de chaque époque

    # Mesure du temps d'entraînement
    start_time = time.time()   # Enregistre l'heure de début

    # Boucle d'entraînement sur le nombre d'époques spécifié
    for epoch in range(num_epochs):
        model.train()          # Mise en mode entraînement
        total_loss = 0         # Initialisation de la perte totale pour l'époque

        # Boucle sur les batches de données
        for seq1, lengths1, seq2, lengths2, labels in train_loader:
            # Déplacement des données sur le device (GPU ou CPU)
            seq1, lengths1 = seq1.to(device), lengths1.to(device)
            seq2, lengths2 = seq2.to(device), lengths2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Réinitialisation des gradients
            # Passage avant dans le modèle
            output1, output2 = model(seq1, lengths1, seq2, lengths2)
            # Calcul de la perte
            loss = criterion(output1, output2, labels)
            # Backpropagation pour calculer les gradients
            loss.backward()
            # Mise à jour des poids du modèle
            optimizer.step()

            # Accumulation de la perte
            total_loss += loss.item()

        # Calcul de la perte moyenne pour l'époque
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)  # Ajout de la perte moyenne à la liste des pertes
        logger.info(f"Époque {epoch+1}/{num_epochs}, Perte moyenne : {avg_loss:.4f}")

        # Vérification pour l'early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss          # Mise à jour de la meilleure perte
            patience_counter = 0          # Réinitialisation du compteur de patience
            # Sauvegarde du meilleur modèle
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1         # Incrémentation du compteur de patience
            # Si le compteur atteint le seuil de patience, on arrête l'entraînement
            if patience_counter >= patience:
                logger.info(f"Arrêt anticipé : la perte moyenne n'a pas diminué depuis {patience} époques.")
                break  # Sortie de la boucle d'entraînement

    # Mesure du temps total d'entraînement
    end_time = time.time()  # Enregistre l'heure de fin
    total_time = end_time - start_time  # Calcul du temps total

    # Statistiques finales
    total_epochs = epoch + 1  # Nombre total d'époques effectuées
    final_loss = losses[-1]   # Perte finale moyenne
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Nombre total de paramètres entraînables

    # Log des statistiques finales
    logger.info("Entraînement terminé.")
    logger.info(f"Nombre total d'époques effectuées : {total_epochs}")
    logger.info(f"Temps total d'entraînement : {total_time:.2f} secondes")
    logger.info(f"Perte finale moyenne : {final_loss:.4f}")
    logger.info(f"Nombre total de paramètres du modèle : {num_parameters}")

    return model, losses  # Retourne le modèle entraîné et la liste des pertes

# Fonction pour évaluer la similarité entre deux séquences avec le modèle entraîné
def evaluate_similarity(
    model: SiameseLSTM,
    idxs1: List[int],
    idxs2: List[int]
) -> float:
    model.eval()  # Mise en mode évaluation
    with torch.no_grad():  # Désactive le calcul des gradients
        # Préparation des séquences
        torch1 = torch.tensor(idxs1, dtype=torch.long)
        seq1 = torch1.unsqueeze(0)  # Ajout d'une dimension batch
        lengths1 = torch.tensor([len(idxs1)])

        torch2 = torch.tensor(idxs2, dtype=torch.long)
        seq2 = torch2.unsqueeze(0)
        lengths2 = torch.tensor([len(idxs2)])

        # Déplacement des données sur le device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq1, lengths1 = seq1.to(device), lengths1.to(device)
        seq2, lengths2 = seq2.to(device), lengths2.to(device)

        # Passage avant dans le modèle
        output1, output2 = model(seq1, lengths1, seq2, lengths2)
        # Calcul de la similarité cosinus entre les représentations
        cosine_similarity = F.cosine_similarity(output1, output2)
        # Ajustement de la similarité pour qu'elle soit entre 0 et 1
        similarity = (cosine_similarity.item() + 1) / 2
        return similarity  # Retourne le score de similarité
