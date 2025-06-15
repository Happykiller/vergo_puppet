# app\neural_network\nn_embedding.py
import time
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader  # type: ignore

from app.common import format_time 
from app.services.logger import logger

# ------------------ MODEL ------------------

class UniversalEmbeddingModel(nn.Module):
    """
    Universal sentence embedding model with similarity prediction using MSELoss.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, lstm_hidden_dim: int = 128):
        super().__init__()
        self.args = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "lstm_hidden_dim": lstm_hidden_dim
        }
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(lstm_hidden_dim * 2, 128)  # Project to embedding space

    def encode(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(seq)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        emb = self.linear(h)
        return F.tanh(emb)  # (batch, 128)

    def forward(self, seq1, len1, seq2, len2):
        emb1 = self.encode(seq1, len1)
        emb2 = self.encode(seq2, len2)
        similarity = F.cosine_similarity(emb1, emb2, dim=1)  # (batch,)
        similarity = (similarity + 1) / 2  # Normalize to [0, 1]
        return similarity

# ------------------ DATASET ------------------

class PairDataset(Dataset):
    """
    Custom Dataset for (seq1, seq2, similarity) training pairs.
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
            torch.tensor(ex["similarity"], dtype=torch.float)
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
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniversalEmbeddingModel(vocab_size, embedding_dim, lstm_hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    dataset = PairDataset(trainset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[train_embedding_model] Total parameters: {total_params}")

    model.train()
    start_time = time.time()
    losses = []
    best_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (seq1, len1, seq2, len2, labels) in enumerate(dataloader):
            seq1, len1 = seq1.to(device), len1.to(device)
            seq2, len2 = seq2.to(device), len2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(seq1, len1, seq2, len2)  # Output: (batch,)
            loss = criterion(preds, labels)
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step()
        logger.info(f"[train_embedding_model] Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            logger.info(f"[train_embedding_model] → New best model (loss={best_loss:.6f}) saved in memory.")

    if save_path and best_model_state:
        torch.save(best_model_state, save_path)
        logger.info(f"[train_embedding_model] Best model saved to {save_path} with (loss={best_loss:.6f})")

    duration = format_time(time.time() - start_time) if 'format_time' in globals() else "N/A"
    logger.info("[train_embedding_model] Training complete.")

    return model, {
        "epochs": num_epochs,
        "final_loss": losses[-1],
        "model_path": save_path,
        "total_time": duration,
        "num_parameters": total_params,
    }
