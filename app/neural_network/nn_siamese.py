#app\neural_network\nn_siamese.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time  # For measuring training time
from app.services.logger import logger  # Import custom logger

# Siamese LSTM model definition
class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        """
        Initializes a Siamese LSTM model.
        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimension of the embedding layer.
        :param hidden_dim: Dimension of LSTM hidden states.
        """
        super(SiameseLSTM, self).__init__()
        # Embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM to encode sequences
        self.lstm = nn.LSTM(
            embedding_dim,      # Input embedding dimension
            hidden_dim,         # LSTM hidden dimension
            batch_first=True,   # Batch is the first dimension
            bidirectional=False # Unidirectional LSTM
        )

    def forward_once(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        Forward pass for a single sequence.
        :param x: Input sequence tensor.
        :param lengths: Lengths of sequences for padding handling.
        :return: Encoded sequence representation.
        """
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),      # Sequence lengths
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output = hidden[-1]  # Use last hidden state as sequence representation
        return output

    def forward(self, input1: torch.Tensor, lengths1: torch.Tensor,
                input2: torch.Tensor, lengths2: torch.Tensor):
        # Encode both sequences
        output1 = self.forward_once(input1, lengths1)
        output2 = self.forward_once(input2, lengths2)
        return output1, output2

# Custom loss function for Siamese model
class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # Mean Squared Error for similarity comparison

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor):
        # Compute cosine similarity between sequence representations
        cosine_similarity = F.cosine_similarity(output1, output2)
        similarity = (cosine_similarity + 1) / 2  # Normalize to [0, 1]
        loss = self.mse_loss(similarity, label)
        return loss

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
    patience: int = 10,
    best_model_path: str = 'best_model.pth'
):
    dataset = SimilarityDataset(training_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = SiameseLSTM(vocab_size, embedding_dim, hidden_dim)
    criterion = SimilarityLoss()
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
            output1, output2 = model(seq1, lengths1, seq2, lengths2)
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        logger.debug(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.8f}")
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
    model.eval()
    with torch.no_grad():
        torch1 = torch.tensor(idxs1, dtype=torch.long)
        seq1 = torch1.unsqueeze(0)
        lengths1 = torch.tensor([len(idxs1)])

        torch2 = torch.tensor(idxs2, dtype=torch.long)
        seq2 = torch2.unsqueeze(0)
        lengths2 = torch.tensor([len(idxs2)])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq1, lengths1 = seq1.to(device), lengths1.to(device)
        seq2, lengths2 = seq2.to(device), lengths2.to(device)

        output1, output2 = model(seq1, lengths1, seq2, lengths2)
        cosine_similarity = F.cosine_similarity(output1, output2)
        similarity = (cosine_similarity.item() + 1) / 2
        return similarity
