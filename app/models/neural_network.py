import torch
import torch.nn as nn
import torch.optim as optim

from app.commons.commons import pad_vector

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
    try:
        input_size = vector_size  # Taille maximale du vecteur d'entrée
        hidden_size = 128  # Taille de la couche cachée
        output_size = vector_size  # Taille de sortie égale à celle du vecteur
        model = SimpleNN(input_size, hidden_size, output_size)

        # Critère de perte et optimiseur
        criterion = nn.MSELoss()  # Erreur quadratique moyenne pour minimiser la différence entre vecteurs
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Convertir les données en tenseurs PyTorch
        # Extrait le premier élément de chaque paire (input vector) dans train_data pour créer la liste des vecteurs d'entrée
        inputs = torch.Tensor([pad_vector(pair[0], vector_size) for pair in train_data])
        # Extrait le second élément de chaque paire (input vector) dans train_data pour créer la liste des vecteurs de destination
        targets = torch.Tensor([pad_vector(pair[1], vector_size) for pair in train_data])

        # Afficher les poids initiaux des couches
        #print("\n[DEBUG] Poids et biais initiaux:")
        #print(f"Poids fc1: {model.fc1.weight.data}")
        #print(f"Biais fc1: {model.fc1.bias.data}")
        #print(f"Poids fc2: {model.fc2.weight.data}")
        #print(f"Biais fc2: {model.fc2.bias.data}")

    except IndexError as e:
        raise ValueError(f"Erreur dans la méthode 'train_model_nn': Problème avec les indices dans les données d'entraînement. Vérifiez que les tuples (input, target) sont correctement formatés. Détails de l'erreur : {str(e)}") from e

    except TypeError as e:
        raise ValueError(f"Erreur dans la méthode 'train_model_nn': Type de données incorrect. Vérifiez que 'train_data' est une liste de tuples (input, target) et que 'vector_size' est un entier. Détails de l'erreur : {str(e)}") from e

    except Exception as e:
        raise RuntimeError(f"Erreur inattendue dans la méthode 'train_model_nn': {str(e)}") from e

    # Entraînement du modèle
    try:
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
                
    except RuntimeError as e:
        raise RuntimeError(f"Erreur lors de l'entraînement du modèle dans la méthode 'train_model_nn': {str(e)}") from e

    except Exception as e:
        raise RuntimeError(f"Erreur inattendue lors de l'entraînement dans la méthode 'train_model_nn': {str(e)}") from e

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
