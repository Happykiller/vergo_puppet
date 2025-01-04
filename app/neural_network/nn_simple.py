#app\neural_network\nn_simple.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from app.services.logger import logger

# Definition of the enhanced SimpleNN model with multiple layers and dropout for regularization
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the SimpleNN model with:
        - input_size: size of the input vector
        - hidden_size: number of neurons in the hidden layer
        - output_size: size of the output vector
        """
        super(SimpleNN, self).__init__()
        # Store the initialization parameters for later access
        self.args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
        }
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # ReLU activation after the first layer
        self.relu1 = nn.ReLU()
        # Second fully connected layer to add depth to the network
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # ReLU activation after the second layer
        self.relu2 = nn.ReLU()
        # Third fully connected layer (output layer)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # Dropout layer to reduce overfitting (50% of neurons randomly disabled)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass function:
        - Takes an input vector x and passes it through the network layers.
        - Applies ReLU activations and dropout for regularization.
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)  # Apply dropout after the first hidden layer
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)  # No activation function after the final layer
        return out

# Function to train the SimpleNN model with early stopping
def train_model_nn(features_processed, targets_standardized, input_size, epochs=3000, learning_rate=0.001, patience=10, improvement_threshold=0.00001):
    """
    Trains the neural network model with the provided features and targets.
    """
    try:
        # Initialize model dimensions
        hidden_size = 128  # Hidden layer size
        output_size = 1  # Output size is 1 (predicting a single value)

        # Create the SimpleNN model
        model = SimpleNN(input_size, hidden_size, output_size)

        # Mean Squared Error Loss function
        criterion = nn.MSELoss()
        # Adam optimizer with initial learning rate
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Convert inputs and targets to PyTorch tensors
        inputs_tensor = torch.tensor(features_processed, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_standardized, dtype=torch.float32).unsqueeze(1)

        # List to store the loss at each epoch
        losses = []
        start_time = time.time()

        best_loss = float('inf')  # Initialize best loss to a high value
        epochs_without_improvement = 0  # Counter for early stopping

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs_tensor)  # Forward pass
            loss = criterion(outputs, targets_tensor)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            current_loss = loss.item()
            losses.append(current_loss)

            # Log loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss}")

            # Check for improvement in loss
            if current_loss < best_loss - improvement_threshold:
                best_loss = current_loss
                epochs_without_improvement = 0  # Reset if loss improves
            else:
                epochs_without_improvement += 1  # Increment if no improvement

            # Stop training if no improvement for specified patience
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}. Optimal loss achieved: {best_loss:.6f}")
                break

        # Calculate total training time
        total_training_time = time.time() - start_time
        total_parameters = sum(p.numel() for p in model.parameters())

        # Loss statistics
        avg_loss = sum(losses) / len(losses)
        min_loss = min(losses)
        max_loss = max(losses)

        # Log training statistics
        logger.info(f"Total training time: {total_training_time:.2f} seconds")
        logger.info(f"Total number of parameters: {total_parameters}")
        logger.info(f"Average loss: {avg_loss}")
        logger.info(f"Minimum loss: {min_loss}")
        logger.info(f"Maximum loss: {max_loss}")
        logger.info(f"Final loss after {len(losses)} epochs: {losses[-1]}")

        return model, losses

    except RuntimeError as e:
        raise RuntimeError(f"Error during model training: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during training: {str(e)}") from e

# Function to make a prediction with the SimpleNN model
def predict(model, input_processed, targets_mean, targets_std):
    try:
        # Convert input to tensor
        input_tensor = torch.tensor(input_processed, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Convert the output tensor to a scalar value
        predicted_standardized = output_tensor.item()
        
        # Un-standardize the prediction
        predicted_price = predicted_standardized * targets_std + targets_mean
        
        return float(predicted_price)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
