#app\neural_network\nn_gru.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

from app.services.logger import logger

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate=0.5):
        """
        Classification model based on a GRU with dropout regularization.
        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimension of the embeddings.
        :param hidden_dim: Dimension of GRU hidden states.
        :param num_classes: Number of target classes.
        :param dropout_rate: Dropout rate for regularization.
        """
        super(GRUClassifier, self).__init__()
        # Store the initialization parameters for later access
        self.args = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
        }
        # Embedding layer to convert word indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # GRU layer to capture sequential dependencies
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, num_classes)
        # LogSoftmax activation to output log-probabilities
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Sequence of word indices (batch_size x seq_length).
        :return: Log-probabilities for each class (batch_size x num_classes).
        """
        # Convert word indices to embeddings
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        # Pass through the GRU layer
        _, hidden = self.gru(embedded)  # hidden shape: (1, batch_size, hidden_dim)
        # Apply dropout to the hidden state
        hidden = self.dropout(hidden.squeeze(0))  # Shape: (batch_size, hidden_dim)
        # Pass through the fully connected layer
        output = self.fc(hidden)  # Shape: (batch_size, num_classes)
        # Apply the LogSoftmax activation function
        output = self.log_softmax(output)
        return output

def train_gru(vocab_size, num_classes, sequences, labels):
    """
    Train the GRU model with dropout and tuned hyperparameters.
    :param vocab_size: Size of the vocabulary.
    :param num_classes: Number of target classes.
    :param sequences: Training sequences tensor.
    :param labels: Corresponding labels tensor.
    :return: Trained model and training statistics.
    """
    start_time = time.time()
    losses = []  # List to store the loss for each epoch
    
    # --- Preprocessing Step: Pre-pad the sequences ---
    # For example, we set a fixed maximum length (you can adjust this according to the distribution of your sequences)
    fixed_max_len = 50  # or compute it based on your data, e.g. max(len(seq) for seq in sequences)
    sequences = pre_pad_sequences(sequences, max_len=fixed_max_len, padding_value=0)

    # Convert input data to tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # Model hyperparameters
    embedding_dim = 128  # Embedding dimension
    hidden_dim = 256     # GRU hidden state dimension
    num_epochs = 1000      # Number of training epochs
    batch_size = 16      # Batch size for updates
    learning_rate = 0.0001  # Learning rate for stable convergence
    dropout_rate = 0.5   # Dropout rate for regularization

    # Initialize the GRU model with Dropout
    model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate)

    # Loss function and optimizer
    criterion = nn.NLLLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # DataLoader to manage batches
    dataset = torch.utils.data.TensorDataset(sequences, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop with early stopping mechanism
    best_loss = float('inf')
    patience = 5  # Epochs without improvement before stopping
    epochs_without_improvement = 0
    best_model_state = None  # To save the best model state
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for batch_sequences, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimizer update
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        logger.debug(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.8f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()  # Save best model state
        else:
            epochs_without_improvement += 1
        
        # Stop if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            logger.info("Early stopping due to lack of loss improvement.")
            break

    # Load the best model state if it exists
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Calculate total training time
    total_training_time = time.time() - start_time
    total_parameters = sum(p.numel() for p in model.parameters())

    # Prepare training report
    training_stats = {
        "total_training_time": total_training_time,
        "total_parameters": total_parameters,
        "min_loss": min(losses),
        "max_loss": max(losses),
        "final_loss": losses[-1] if losses else None,
        "epochs_run": len(losses),
        "early_stopping_triggered": epochs_without_improvement >= patience,
        "best_loss": best_loss,
    }

    # Log training statistics
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    logger.info(f"Total parameters: {total_parameters}")
    logger.info(f"Max loss: {training_stats['max_loss']}")
    logger.info(f"Min loss: {training_stats['min_loss']}")
    logger.info(f"Final loss: {training_stats['final_loss']}")
    logger.info(f"Epochs run: {training_stats['epochs_run']}")
    if training_stats["early_stopping_triggered"]:
        logger.info("Early stopping was triggered.")
    logger.info(f"Best loss: {training_stats['best_loss']}")

    return model, training_stats
        
def predict(nn_model, input):
    """
    Load the trained GRU model and perform prediction on processed input.
    :param nn_model: Trained GRU model.
    :param input: Preprocessed input sequence.
    :return: Index of the predicted class.
    """
    input_processed = torch.tensor([input], dtype=torch.long)
    
    # Prediction
    with torch.no_grad():
        outputs = nn_model(input_processed)
        predicted_idx = torch.argmax(outputs, dim=1).item()
    
    return predicted_idx

def pre_pad_sequences(sequences, max_len=None, padding_value=0):
    """
    Pre-pad (or truncate) a list of token sequences so that all sequences have the same length.
    
    :param sequences: List of sequences (each sequence is a list of integers).
    :param max_len: Desired maximum length. If None, the maximum length from the sequences is used.
    :param padding_value: Value to use for padding (0 in your case, since padding_idx=0 in the embedding layer).
    :return: List of pre-padded sequences.
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            # Append padding tokens at the end (post-padding)
            padded_seq = seq + [padding_value] * (max_len - len(seq))
        else:
            # Truncate the sequence if necessary
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return padded_sequences
