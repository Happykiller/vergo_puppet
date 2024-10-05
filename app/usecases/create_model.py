from app.repositories.memory import model_exists, save_model
from fastapi import HTTPException  # type: ignore
from typing import List

# Importer la fonction de conversion
from app.usecases.tokens_to_indices import tokens_to_indices  # Ajuster le chemin si nécessaire

def create_model(name: str, dictionary: List[List[str]], glossary: List[str], neural_network_type="SimpleNN"):
    if model_exists(name):
        raise HTTPException(status_code=400, detail="Model already exists")
    
    # Vérifier si dictionary ou glossary sont None
    if dictionary is None:
        raise HTTPException(status_code=400, detail="Dictionary cannot be None")
    if glossary is None:
        raise HTTPException(status_code=400, detail="Glossary cannot be None")
    
    # Vérifier si dictionary ou glossary sont vides
    if len(dictionary) == 0:
        raise HTTPException(status_code=400, detail="Dictionary cannot be empty")
    if len(glossary) == 0:
        raise HTTPException(status_code=400, detail="Glossary cannot be empty")

    # Ajouter une valeur vierge au début du glossaire
    glossary = [""] + ["UNK"] + glossary

    # Transformer chaque liste de tokens en liste d'indices
    indexed_dictionary = [
        tokens_to_indices(tokens, glossary) for tokens in dictionary
    ]

    # Enregistrer le modèle avec le glossaire et le dictionnaire d'indices
    model_data = {
        "dictionary": dictionary,
        "indexed_dictionary": indexed_dictionary, # Dictionnaire transformé avec les indices
        "glossary": glossary,  # Glossaire original
        "neural_network_type": neural_network_type # Enregistrement du type de modèle
    }
    save_model(name, model_data)

    return {"status": "model created", "model_name": name}
