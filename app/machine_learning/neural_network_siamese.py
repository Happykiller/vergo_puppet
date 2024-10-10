# neural_network_siamese.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super(SiameseLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=False
        )

    def forward_once(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)
        # Pack la séquence pour gérer les longueurs variables
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # On utilise le dernier état caché
        output = hidden[-1]
        return output

    def forward(self, input1: torch.Tensor, lengths1: torch.Tensor, input2: torch.Tensor, lengths2: torch.Tensor):
        output1 = self.forward_once(input1, lengths1)
        output2 = self.forward_once(input2, lengths2)
        return output1, output2

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor):
        # Calcul de la similarité cosinus entre les représentations
        cosine_similarity = F.cosine_similarity(output1, output2)
        # Ajuster la similarité pour qu'elle soit entre 0 et 1
        similarity = (cosine_similarity + 1) / 2  # Transforme [-1,1] en [0,1]
        # Calcul de la perte
        loss = self.mse_loss(similarity, label)
        return loss

# Fonction pour convertir une séquence de tokens en indices
def tokens_to_indices(tokens: List[str], word2idx: dict) -> List[int]:
    return [word2idx.get(token, 0) for token in tokens]

# Fonction pour créer des paires de séquences avec leur longueur
def prepare_sequence(seq: List[str], word2idx: dict) -> torch.Tensor:
    idxs = tokens_to_indices(seq, word2idx)
    return torch.tensor(idxs, dtype=torch.long)

# Dataset personnalisé pour les paires de similarité
class SimilarityDataset(Dataset):
    def __init__(self, data: List[Tuple[List[str], List[str], float]], word2idx: dict):
        self.pairs = []
        self.labels = []
        for seq1, seq2, label in data:
            self.pairs.append(
                (prepare_sequence(seq1, word2idx), prepare_sequence(seq2, word2idx))
            )
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq1, seq2 = self.pairs[idx]
        label = self.labels[idx]
        return seq1, seq2, label

# Fonction pour le collate_fn afin de gérer les batchs avec séquences de longueur variable
def collate_fn(data):
    seq1_list, seq2_list, label_list = zip(*data)
    # Séquences 1
    lengths1 = [len(seq) for seq in seq1_list]
    seq1_padded = nn.utils.rnn.pad_sequence(
        seq1_list, batch_first=True, padding_value=0
    )
    # Séquences 2
    lengths2 = [len(seq) for seq in seq2_list]
    seq2_padded = nn.utils.rnn.pad_sequence(
        seq2_list, batch_first=True, padding_value=0
    )
    # Labels
    labels = torch.tensor(label_list, dtype=torch.float)
    return (
        seq1_padded,
        torch.tensor(lengths1),
        seq2_padded,
        torch.tensor(lengths2),
        labels,
    )

# Fonction pour entraîner le modèle Siamese LSTM
def train_siamese_model_nn(
    training_data: List[Tuple[List[str], List[str], float]],
    word2idx: dict,
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    batch_size: int = 32
):
    # Création du dataset et du DataLoader
    dataset = SimilarityDataset(training_data, word2idx)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Initialisation du modèle, de la perte et de l'optimiseur
    model = SiameseLSTM(vocab_size, embedding_dim, hidden_dim)
    criterion = SimilarityLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Déplacement du modèle sur le device (CPU ou GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Boucle d'entraînement
    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for seq1, lengths1, seq2, lengths2, labels in train_loader:
            seq1, lengths1 = seq1.to(device), lengths1.to(device)
            seq2, lengths2 = seq2.to(device), lengths2.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output1, output2 = model(seq1, lengths1, seq2, lengths2)
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Époque {epoch+1}/{num_epochs}, Perte moyenne : {avg_loss:.4f}")

    return model, losses

# Fonction pour évaluer la similarité entre deux séquences
def evaluate_similarity(
    model: SiameseLSTM,
    seq1_tokens: List[str],
    seq2_tokens: List[str],
    word2idx: dict
) -> float:
    model.eval()
    with torch.no_grad():
        seq1 = prepare_sequence(seq1_tokens, word2idx).unsqueeze(0)
        lengths1 = torch.tensor([len(seq1_tokens)])
        seq2 = prepare_sequence(seq2_tokens, word2idx).unsqueeze(0)
        lengths2 = torch.tensor([len(seq2_tokens)])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq1, lengths1 = seq1.to(device), lengths1.to(device)
        seq2, lengths2 = seq2.to(device), lengths2.to(device)
        output1, output2 = model(seq1, lengths1, seq2, lengths2)
        cosine_similarity = F.cosine_similarity(output1, output2)
        similarity = (cosine_similarity.item() + 1) / 2  # Transforme [-1,1] en [0,1]
        return similarity
