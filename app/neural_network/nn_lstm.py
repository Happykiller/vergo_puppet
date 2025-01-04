#app\neural_network\nn_lstm.py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.services.logger import logger

class LSTMNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Initialize an LSTM-based neural network for regression tasks.
        :param input_size: Number of input features per time step.
        :param hidden_size: Number of hidden units in the LSTM layer.
        """
        super(LSTMNN, self).__init__()
        # Store the initialization parameters for later access
        self.args = {
            "input_size": input_size,
            "hidden_size": hidden_size
        }
        # LSTM layer to capture temporal dependencies in the sequence
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Fully connected layer for outputting final predictions
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Perform forward pass.
        :param x: Input tensor of shape (batch_size, sequence_length, input_size).
        :return: Output tensor of shape (batch_size, 1).
        """
        # Pass through the LSTM layer; only take the output from the last time step
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Select the last output for each sequence
        out = self.fc(lstm_out)  # Pass through the fully connected layer
        return out

def train_nn_lstm(X_train, y_train, epochs=20, learning_rate=0.001, patience=10):
    """
    Train the LSTM model with early stopping based on validation loss.
    :param X_train: Training data, shape (num_samples, sequence_length, input_size).
    :param y_train: Target values, shape (num_samples, 1).
    :param epochs: Number of training epochs.
    :param learning_rate: Learning rate for the optimizer.
    :param patience: Number of epochs to wait for improvement before stopping.
    :return: Trained LSTM model.
    """
    start_time = time.time()

    # Convert training data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    
    # Define model parameters
    input_size = X_train.shape[2]  # Number of input features per time step
    hidden_size = 64  # Adjust hidden size as needed
    
    nn_model = LSTMNN(input_size, hidden_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)

    # Variables for early stopping
    best_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    # Training loop with early stopping
    for epoch in range(epochs):
        nn_model.train()  # Set model to training mode
        optimizer.zero_grad()
        outputs = nn_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        # Check for improvement
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch + 1
            patience_counter = 0
            # Save the best model state
            best_model_state = nn_model.state_dict()
        else:
            patience_counter += 1
        
        # Log every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"No improvement for {patience} consecutive epochs. Early stopping at epoch {epoch+1}.")
            break

    # Load the best model state if it exists
    if best_model_state is not None:
        nn_model.load_state_dict(best_model_state)
        logger.info(f"Best model achieved at epoch {best_epoch} with validation loss of {best_loss:.6f}")
    else:
        logger.warning("No improvement observed during training.")

    # Calculate total training time
    total_training_time = time.time() - start_time

    # Log training statistics
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    
    return nn_model

def predict_nn_lstm(nn_model, X_input):
    """
    Use the trained LSTM model to make predictions on new data.
    :param nn_model: Trained LSTM model.
    :param X_input: Input data, shape (batch_size, sequence_length, input_size).
    :return: Predictions as a numpy array.
    """
    nn_model.eval()  # Set model to evaluation mode

    # Convert input data to PyTorch tensor
    X_input_tensor = torch.Tensor(X_input)

    # Disable gradient calculation to save time and memory
    with torch.no_grad():
        predictions = nn_model(X_input_tensor)

    # Convert predictions to numpy array
    predictions_np = predictions.numpy().reshape(-1)

    return predictions_np
