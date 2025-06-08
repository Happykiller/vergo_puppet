# app\neural_network\nn_siamese.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from app.common import format_time
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
        Initialize an advanced Siamese LSTM model.
        :param vocab_size: Vocabulary size.
        :param embedding_dim: Dimension of the embedding layer.
        :param hidden_dim: Dimension of the LSTM hidden states.
        :param num_layers: Number of LSTM layers.
        :param bidirectional: Whether to use a bidirectional LSTM.
        :param dropout: Dropout rate.
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
        
        # For multiple layers, dropout is automatically applied between layers except the last
        self.lstm = nn.LSTM(
            embedding_dim,      # Input embedding dimension
            hidden_dim,         # LSTM hidden dimension
            num_layers=2,
            batch_first=True,   # Batch is the first dimension
            bidirectional=True # Bidirectional LSTM
        )
        # Dropout layer for regularization applied after LSTM encoding
        self.dropout = nn.Dropout(0.5)
        fc_input_dim = hidden_dim * 4
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward_once(self, x, lengths):
        """
        Perform the forward pass for a single sequence.
        :param x: Input tensor for the sequence.
        :param lengths: Sequence lengths used for padding management.
        :return: Sequence representation obtained by concatenating the final states of both directions.
        """
        # Convert indices to dense vectors
        embedded = self.embedding(x)
        # Handle variable-length sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),      # Sequence lengths
            batch_first=True,
            enforce_sorted=False
        )
        # Pass through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        # For a two-layer bidirectional LSTM the shape is (4, batch, hidden_dim)
        # Retrieve the final states from the last layer:
        # - hidden[-2] corresponds to the forward state
        # - hidden[-1] corresponds to the backward state
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        # Concatenate the states to obtain a representation of dimension hidden_dim*2
        output = torch.cat((forward_hidden, backward_hidden), dim=1)
        return output

    def forward(self, input1: torch.Tensor, lengths1: torch.Tensor,
                input2: torch.Tensor, lengths2: torch.Tensor):
        """
        Forward pass for both sequences. Each sequence is encoded, the two representations are
        concatenated, dropout is applied and finally the fully connected layer generates the
        similarity score.
        """
        # Encode each sequence
        output1 = self.forward_once(input1, lengths1)
        output2 = self.forward_once(input2, lengths2)
        # Apply dropout to each output for regularization
        output1 = self.dropout(output1)
        output2 = self.dropout(output2)
        # Compute cosine similarity between the two representations
        cosine_similarity = F.cosine_similarity(output1, output2)
        # Normalize cosine similarity from [-1, 1] to [0, 1]
        similarity = torch.clamp(cosine_similarity, min=0)
        return similarity.unsqueeze(1)  # Ensure output shape is (batch, 1)

# Custom loss function for Siamese model
class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # Mean Squared Error for similarity comparison

    def forward(self, similarity_score: torch.Tensor, label: torch.Tensor):
        """
        Compute the loss by comparing the predicted similarity score to the ground truth label.
        :param similarity_score: Predicted similarity score from the model.
        :param label: Similarity label expected in the range [0, 1].
        :return: Loss value.
        """
        loss = self.mse_loss(similarity_score, label)
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
    """
    Pad sequence pairs and return tensors of sequences, lengths and labels.

    :param data: Iterable of ``(seq1, seq2, label)`` tuples.
    :return: Padded sequences and associated length and label tensors.
    """
    seq1_list, seq2_list, label_list = zip(*data)
    lengths1 = [len(seq) for seq in seq1_list]
    seq1_padded = nn.utils.rnn.pad_sequence(seq1_list, batch_first=True, padding_value=0)
    lengths2 = [len(seq) for seq in seq2_list]
    seq2_padded = nn.utils.rnn.pad_sequence(seq2_list, batch_first=True, padding_value=0)
    labels = torch.tensor(label_list, dtype=torch.float).unsqueeze(1)
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
    """
    Train a Siamese LSTM network on sequence pairs with early stopping.

    :param training_data: List of ``(indices1, indices2, similarity)`` tuples.
    :param vocab_size: Size of the token vocabulary.
    :param embedding_dim: Dimension of the embedding layer.
    :param hidden_dim: Dimension of the LSTM hidden state.
    :param num_epochs: Maximum number of training epochs.
    :param learning_rate: Learning rate for Adam optimizer.
    :param batch_size: Mini-batch size used during training.
    :param patience: Epochs to wait without improvement before stopping.
    :param best_model_path: Path where the best model will be saved.
    :return: Tuple ``(model, report)`` with the trained model and metrics.
    """
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
            similarity = model(seq1, lengths1, seq2, lengths2)
            loss = criterion(similarity, labels)
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
    logger.info(f"Total training time: {format_time(total_time)}")
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
    """
    Compute the similarity score between two index sequences.

    :param model: Trained Siamese LSTM model.
    :param idxs1: Indices representing the first sequence.
    :param idxs2: Indices representing the second sequence.
    :return: Similarity score rounded to three decimals.
    """
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
