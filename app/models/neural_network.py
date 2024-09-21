import torch
import torch.nn as nn
import torch.optim as optim

# Définition du réseau de neurones
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Fonction pour entraîner le modèle avec des impressions de debug
def train_model_nn(train_data, vector_size, epochs=100, learning_rate=0.001):
    input_size = vector_size
    hidden_size = 128  # Taille de la couche cachée
    output_size = vector_size  # Taille de sortie égale à celle du vecteur
    model = SimpleNN(input_size, hidden_size, output_size)

    # Critère de perte et optimiseur
    criterion = nn.MSELoss()  # Erreur quadratique moyenne pour minimiser la différence entre vecteurs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convertir les données en tenseurs PyTorch
    inputs = torch.Tensor([pair[0] for pair in train_data])
    targets = torch.Tensor([pair[1] for pair in train_data])

    # Afficher les poids initiaux des couches
    #print("\n[DEBUG] Poids et biais initiaux:")
    #print(f"Poids fc1: {model.fc1.weight.data}")
    #print(f"Biais fc1: {model.fc1.bias.data}")
    #print(f"Poids fc2: {model.fc2.weight.data}")
    #print(f"Biais fc2: {model.fc2.bias.data}")

    losses = []  # Stocker la perte à chaque époque

    # Entraîner le modèle
    for epoch in range(epochs):
        optimizer.zero_grad()  # Réinitialiser les gradients
        outputs = model(inputs)  # Propagation avant
        loss = criterion(outputs, targets)  # Calcul de la perte
        loss.backward()  # Rétropropagation pour calculer les gradients
        optimizer.step()  # Mise à jour des poids

        losses.append(loss.item())

        # Afficher la perte toutes les 10 époques pour le suivi
        if (epoch+1) % 10 == 0:
            print(f"[DEBUG] Époque {epoch+1}/{epochs}, Perte: {loss.item()}")

    # Afficher les poids après entraînement
    #print("\n[DEBUG] Poids et biais après entraînement:")
    #print(f"Poids fc1: {model.fc1.weight.data}")
    #print(f"Biais fc1: {model.fc1.bias.data}")
    #print(f"Poids fc2: {model.fc2.weight.data}")
    #print(f"Biais fc2: {model.fc2.bias.data}")

    print(f"[DEBUG] Perte finale: {losses[-1]}")
    return model, losses

# Fonction pour faire une prédiction
def predict(model, input_vector):
    input_tensor = torch.Tensor(input_vector)
    with torch.no_grad():
        output_vector = model(input_tensor)
    
    print(f"[DEBUG] Vecteur de sortie prédit : {output_vector}")
    return output_vector
