# app\neural_network\nn_embedding.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader

# ------------------ MODEL ------------------

class UniversalEmbeddingModel(nn.Module):
    """
    Simple universal sentence embedding model using Embedding + LSTM pooling.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, lstm_hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(lstm_hidden_dim * 2, 128)  # Output dimension can be customized

    def encode(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of sequences into fixed-size embeddings.
        :param seq: (batch, seq_len) token indices
        :param lengths: (batch,) real sequence lengths
        :return: (batch, embedding_dim) sentence embedding
        """
        emb = self.embedding(seq)  # (batch, seq_len, embedding_dim)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # Concatenate final forward & backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        return self.linear(h)  # (batch, output_dim)

    def forward(self, seq1, len1, seq2, len2):
        """
        Forward pass for a pair of sequences.
        Returns their embeddings.
        """
        emb1 = self.encode(seq1, len1)
        emb2 = self.encode(seq2, len2)
        return emb1, emb2

# ------------------ DATASET ------------------

class PairDataset(Dataset):
    """
    Custom Dataset for (seq1, seq2, label) training pairs.
    """
    def __init__(self, pairs: List[Dict[str, Any]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ex = self.pairs[idx]
        return (
            torch.tensor(ex["seq1"], dtype=torch.long),
            torch.tensor(ex["seq2"], dtype=torch.long),
            torch.tensor(ex["label"], dtype=torch.float)
        )

def pad_collate_fn(batch):
    """
    Collate function to pad sequence batches for DataLoader.
    """
    seq1, seq2, labels = zip(*batch)
    seq1_lengths = torch.tensor([len(s) for s in seq1])
    seq2_lengths = torch.tensor([len(s) for s in seq2])
    seq1_padded = nn.utils.rnn.pad_sequence(seq1, batch_first=True, padding_value=0)
    seq2_padded = nn.utils.rnn.pad_sequence(seq2, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return seq1_padded, seq1_lengths, seq2_padded, seq2_lengths, labels

# ------------------ TRAINING LOOP ------------------

def train_embedding_model(
    trainset: List[Dict[str, Any]],
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_hidden_dim: int = 128,
    batch_size: int = 128,
    num_epochs: int = 2,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trains the UniversalEmbeddingModel on provided training data.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniversalEmbeddingModel(vocab_size, embedding_dim, lstm_hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)

    dataset = PairDataset(trainset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for seq1, len1, seq2, len2, labels in dataloader:
            seq1, len1 = seq1.to(device), len1.to(device)
            seq2, len2 = seq2.to(device), len2.to(device)
            labels = labels.to(device) * 2 - 1  # convert 1.0/0.0 to 1/-1 for CosineEmbeddingLoss

            optimizer.zero_grad()
            emb1, emb2 = model(seq1, len1, seq2, len2)
            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}")

    # Save the model weights (if required)
    if save_path:
        torch.save(model.state_dict(), save_path)

    return model, {
        "epochs": num_epochs,
        "final_loss": total_loss / len(dataloader),
        "model_path": save_path
    }

# ------------------ UTILITIES ------------------

def load_embedding_model(weights_path: str, vocab_size: int, embedding_dim: int = 128, lstm_hidden_dim: int = 128) -> UniversalEmbeddingModel:
    """
    Loads a trained UniversalEmbeddingModel from file.
    """
    model = UniversalEmbeddingModel(vocab_size, embedding_dim, lstm_hidden_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model
