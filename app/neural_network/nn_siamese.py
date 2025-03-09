# app\neural_network\nn_siamese.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import torch.nn.functional as F
import time  # For measuring training time
from torch.utils.data import Dataset, DataLoader

from app.services.logger import logger  # Import custom logger

# Siamese LSTM model definition
class SiameseLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.5
    ):
        """
        Initialise un modèle Siamese LSTM évolué.
        :param vocab_size: Taille du vocabulaire.
        :param embedding_dim: Dimension de la couche d'embedding.
        :param hidden_dim: Dimension des états cachés du LSTM.
        :param num_layers: Nombre de couches LSTM.
        :param bidirectional: Utiliser un LSTM bidirectionnel ou non.
        :param dropout: Taux de dropout.
        """
        super().__init__()
        
        self.args = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout
        }
        
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Pour plusieurs couches, le dropout est appliqué automatiquement entre les couches (sauf la dernière)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        # Si bidirectionnel, la représentation d'un document est de dimension hidden_dim * 2
        fc_input_dim = (hidden_dim * 2 if bidirectional else hidden_dim) * 2  # *2 pour la concaténation des deux documents
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward_once(self, x, lengths):
        """
        Passe une séquence dans le LSTM pour obtenir sa représentation.
        """
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        
        if self.bidirectional:
            # hidden a la forme (num_layers * 2, batch, hidden_dim)
            # On récupère les états du dernier niveau : 
            #   - l'état forward est à l'indice -2
            #   - l'état backward est à l'indice -1
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            hidden_concat = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            # Si non bidirectionnel, on prend simplement le dernier état caché
            hidden_concat = hidden[-1]
        
        # Optionnel : appliquer un dropout sur la représentation finale
        return self.dropout(hidden_concat)

    def forward(self, seq1, lengths1, seq2, lengths2):
        emb1 = self.forward_once(seq1, lengths1)
        emb2 = self.forward_once(seq2, lengths2)
        
        # Concaténation des représentations des deux documents
        combined = torch.cat([emb1, emb2], dim=1)
        similarity = torch.sigmoid(self.fc(combined))
        return similarity.squeeze()

# Custom dataset for sequence pairs and similarity labels
class SimilarityDataset(Dataset):
    def __init__(self, data: List[Tuple[List[int], List[int], float]]):
        self.pairs = []
        self.labels = []
        for idxs1, idxs2, label in data:
            torch1 = torch.tensor(idxs1, dtype=torch.long)
            torch2 = torch.tensor(idxs2, dtype=torch.long)
            self.pairs.append((torch1, torch2))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq1, seq2 = self.pairs[idx]
        label = self.labels[idx]
        return seq1, seq2, label

# Collate function for DataLoader to handle variable-length sequences
def collate_fn(data):
    seq1_list, seq2_list, label_list = zip(*data)
    lengths1 = [len(seq) for seq in seq1_list]
    seq1_padded = nn.utils.rnn.pad_sequence(seq1_list, batch_first=True, padding_value=0)
    lengths2 = [len(seq) for seq in seq2_list]
    seq2_padded = nn.utils.rnn.pad_sequence(seq2_list, batch_first=True, padding_value=0)
    labels = torch.tensor(label_list, dtype=torch.float)
    return seq1_padded, torch.tensor(lengths1), seq2_padded, torch.tensor(lengths2), labels

# Function to train the Siamese LSTM model with early stopping and logging
def train_siamese_model_nn(
    training_data: List[Tuple[List[int], List[int], float]],
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    patience: int = 20,
    best_model_path: str = 'best_model.pth'
):
    dataset = SimilarityDataset(training_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = SiameseLSTM(vocab_size, embedding_dim, hidden_dim, 2, True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_loss = float('inf')
    patience_counter = 0
    losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for seq1, lengths1, seq2, lengths2, labels in train_loader:
            seq1, lengths1 = seq1.to(device), lengths1.to(device)
            seq2, lengths2 = seq2.to(device), lengths2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            similarity_pred = model(seq1, lengths1, seq2, lengths2)
            loss = criterion(similarity_pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        logger.debug(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.8f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping: no improvement in {patience} epochs.")
                break

    end_time = time.time()
    total_time = end_time - start_time
    total_epochs = epoch + 1
    final_loss = losses[-1]
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Reload the best model before returning
    logger.info("Reloading the best model with lowest loss.")
    model.load_state_dict(torch.load(best_model_path))

    logger.info("Training complete.")
    logger.info(f"Total epochs: {total_epochs}")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Final average loss: {final_loss:.8f}")
    logger.info(f"Best loss: {best_loss:.8f}")
    logger.info(f"Total model parameters: {num_parameters}")

    report = {
        "total_epochs": total_epochs,
        "possible_epochs": num_epochs,
        "total_time": total_time,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "num_parameters": num_parameters
    }

    return model, report

# Function to evaluate similarity between two sequences with the trained model
def evaluate_similarity(
    model: SiameseLSTM,
    idxs1: List[int],
    idxs2: List[int]
) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        seq1 = torch.tensor(idxs1, dtype=torch.long).unsqueeze(0).to(device)
        lengths1 = torch.tensor([len(idxs1)], dtype=torch.long).to(device)

        seq2 = torch.tensor(idxs2, dtype=torch.long).unsqueeze(0).to(device)
        lengths2 = torch.tensor([len(idxs2)], dtype=torch.long).to(device)

        similarity = model(seq1, lengths1, seq2, lengths2)
        return round(similarity.item(), 3)
