# app\neural_network\nn_siamese2.py
import torch
import random
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 1) Model Definition
# -------------------------
class Siamese2(nn.Module):
    """
    Siamese LSTM model that encodes two sequences of indices and
    outputs an embedding for each. We then compare those embeddings
    to compute a similarity score.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super(Siamese2, self).__init__()
        # Save hyperparameters for reference
        self.args = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
        }
        
        # Embedding layer to learn a dense representation of each index
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # in case we pad sequences with 0
        )
        
        # LSTM for each sequence; we will call forward_once() on each sequence
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        
        # Final projection for each side (e.g. optional FC layer after LSTM)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward_once(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode a single sequence through the embedding + LSTM.
        :param x: Tensor of shape [batch_size, seq_len], containing indices
        :param lengths: 1D Tensor containing sequence lengths for each batch row
        :return: A tensor of shape [batch_size, hidden_dim] (final embedding)
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Pack the sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),  # lengths must be on CPU for pack_padded_sequence
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM
        packed_out, (hidden, cell) = self.lstm(packed)
        # hidden shape: [num_layers, batch_size, hidden_dim] => we take last layer:
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Optional final projection
        out = self.projection(last_hidden)  # [batch_size, hidden_dim]
        return out
    
    def forward(
        self,
        seq1: torch.Tensor, lengths1: torch.Tensor,
        seq2: torch.Tensor, lengths2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for two sequences. Returns the predicted similarity (0..1).
        """
        out1 = self.forward_once(seq1, lengths1)  # [batch_size, hidden_dim]
        out2 = self.forward_once(seq2, lengths2)  # [batch_size, hidden_dim]
        
        # Compute the cosine similarity and normalize it to [0..1]
        # cos_sim in [-1..+1] => transform to (cos_sim + 1)/2 => [0..1]
        cos_sim = F.cosine_similarity(out1, out2, dim=1)  # [batch_size]
        similarity = (cos_sim + 1.0) / 2.0  # shift to [0..1]
        
        return similarity


# -------------------------
# 2) Dataset + Collate
# -------------------------
class Similarity2Dataset(Dataset):
    """
    Custom Dataset that stores triplets: (sequenceA, sequenceB, labelSimilarity).
    Each sequence is a list of indices (int).
    labelSimilarity is a float in [0..1].
    """
    def __init__(self, data: List[Tuple[List[int], List[int], float]]):
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        seqA, seqB, sim = self.data[idx]
        return seqA, seqB, sim

def collate_fn(batch):
    """
    Collate function to handle variable-length sequences for the LSTM.
    The batch is a list of (seqA, seqB, similarity).
    We pad each sequence so they have the same length in the batch.
    Also generate a 'length' tensor for each side.
    """
    seqA_list, seqB_list, sim_list = zip(*batch)
    
    # Convert each list of int to a torch.Tensor, then keep track of lengths
    lengthsA = [len(a) for a in seqA_list]
    lengthsB = [len(b) for b in seqB_list]
    
    # Pad sequences with 0
    paddedA = nn.utils.rnn.pad_sequence(
        [torch.tensor(a, dtype=torch.long) for a in seqA_list],
        batch_first=True,
        padding_value=0
    )
    paddedB = nn.utils.rnn.pad_sequence(
        [torch.tensor(b, dtype=torch.long) for b in seqB_list],
        batch_first=True,
        padding_value=0
    )
    
    # Convert similarity to tensor
    sim_tensor = torch.tensor(sim_list, dtype=torch.float)
    
    return paddedA, torch.tensor(lengthsA), paddedB, torch.tensor(lengthsB), sim_tensor


# -------------------------
# 3) Training Function
# -------------------------
def train_siamese2(
    training_data: List[Tuple[List[int], List[int], float]],
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3
) -> Tuple[Siamese2, dict]:
    """
    Trains a Siamese2 LSTM model on the given training data.
    :param training_data: List of (seqA, seqB, similarity)
    :param vocab_size: Number of unique indices in your "vocabulary"
    :param embedding_dim: Size of each embedding vector
    :param hidden_dim: Number of LSTM hidden units
    :param epochs: Number of training epochs
    :param batch_size: Size of the mini-batches
    :param learning_rate: Initial learning rate for Adam
    :return: A trained Siamese2 model
    """
    # 1) Create dataset + dataloader
    dataset = Similarity2Dataset(training_data)
    
    # 1.1) Informations sur le dataset
    nb_pairs = len(dataset)
    print(f"Nombre total de paires d'entraînement : {nb_pairs}")

    # 1.2) Afficher jusqu'à 10 échantillons aléatoires
    nb_samples = min(nb_pairs, 10)
    random_indices = random.sample(range(nb_pairs), nb_samples)
    print(f"\nQuelques exemples de paires d'entraînement (max 10) :")
    for i, idx in enumerate(random_indices):
        seqA, seqB, sim = dataset[idx]
        print(f"Exemple #{i+1} => seqA={seqA}, seqB={seqB}, simil={sim}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 2) Instantiate the model and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Siamese2(vocab_size, embedding_dim, hidden_dim).to(device)
    
    # 3) Define optimizer + loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # We want to regress predicted similarity to label
    
    losses_per_epoch = []
    
    # 4) Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for seqA, lenA, seqB, lenB, sims in dataloader:
            # Move everything to device
            seqA = seqA.to(device)
            lenA = lenA.to(device)
            seqB = seqB.to(device)
            lenB = lenB.to(device)
            sims = sims.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass => predictions
            preds = model(seqA, lenA, seqB, lenB)  # shape: [batch_size]
            
            # Compute the loss
            loss = criterion(preds, sims)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses_per_epoch.append(avg_loss)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
    training_report = {
        "losses": losses_per_epoch,
        "final_loss": losses_per_epoch[-1] if losses_per_epoch else None,
        "epochs_run": epochs
    }

    return model, training_report


# -------------------------
# 4) Evaluate Function
# -------------------------
def evaluate_similarity_siamese2(
    model: Siamese2,
    idxs1: List[int],
    idxs2: List[int]
) -> float:
    """
    Compute the similarity score between two sequences of indices.
    The model should be in eval mode.
    """
    device = next(model.parameters()).device  # get the device from the model
    
    model.eval()
    with torch.no_grad():
        # Convert lists to Tensors
        seq1_tensor = torch.tensor([idxs1], dtype=torch.long, device=device)
        seq2_tensor = torch.tensor([idxs2], dtype=torch.long, device=device)
        
        len1 = torch.tensor([len(idxs1)], dtype=torch.long, device=device)
        len2 = torch.tensor([len(idxs2)], dtype=torch.long, device=device)
        
        # Forward pass
        score_tensor = model(seq1_tensor, len1, seq2_tensor, len2)  # shape: [1]
        score = score_tensor.item()
    
    return score

# -------------------------
# 5) Example Usage
# -------------------------
if __name__ == "__main__":
    # Suppose we have some small training_data
    # Format: (sequenceA, sequenceB, similarity)
    # similarity is in [0..1]
    toy_data = [
        ([1, 2, 3], [1, 2, 4], 0.8),
        ([5, 6], [8, 6, 5], 0.4),
        ([2, 3, 9], [2, 3, 9], 1.0),
        ([10, 10, 10], [2, 7], 0.0),
        ([3, 4, 5], [3, 4, 6], 0.7),
    ]
    
    # We need vocab_size = max_index+1 if indices start at 0
    # For the sake of example, let's guess the maximum index is 10
    vocab_size = 11  # indices 0..10
    
    # Train the model
    trained_model, _ = train_siamese2(
        training_data=toy_data,
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        epochs=5,
        batch_size=2,
        learning_rate=1e-3
    )
    
    # Evaluate similarity
    seqA = [1, 2, 3]
    seqB = [1, 2, 4]
    sim_score = evaluate_similarity_siamese2(trained_model, seqA, seqB)
    print(f"Similarity score between {seqA} and {seqB} is: {sim_score:.4f}")
    
    # Evaluate similarity
    seqA = [10, 10, 10]
    seqB = [2, 7]
    sim_score = evaluate_similarity_siamese2(trained_model, seqA, seqB)
    print(f"Similarity score between {seqA} and {seqB} is: {sim_score:.4f}")
    
    # Evaluate similarity
    seqA = [2, 3, 9]
    sim_score = evaluate_similarity_siamese2(trained_model, seqA, seqA)
    print(f"Similarity score between {seqA} and {seqA} is: {sim_score:.4f}")
