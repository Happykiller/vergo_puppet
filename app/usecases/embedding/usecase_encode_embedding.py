# app\usecases\embedding\usecase_encode_embedding.py
import torch
import traceback
from pathlib import Path

from app.neural_network.nn_embedding import load_embedding_model
from app.services.logger import logger

def encode_embedding_usecase(
    model_name: str,
    vocab: dict,
    sentence: str,
    embedding_dim: int = 128,
    lstm_hidden_dim: int = 128,
) -> list[float]:
    """
    Encodes a sentence into its embedding vector using a trained model.
    :param model_name: Name of the model to load
    :param vocab: Token-to-index mapping
    :param sentence: The sentence to encode
    :param embedding_dim: Model embedding dimension (default 128)
    :param lstm_hidden_dim: Model LSTM hidden size (default 128)
    :return: List of floats (embedding)
    """
    try:
        # 1. Load the model
        model_path = Path("files") / f"{model_name}_embedding.pt"
        model = load_embedding_model(str(model_path), vocab_size=len(vocab), embedding_dim=embedding_dim, lstm_hidden_dim=lstm_hidden_dim)
        
        # 2. Tokenize and convert to indices
        import re
        def simple_tokenize(text):
            return re.findall(r"\b\w+\b", text.lower())
        indices = [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenize(sentence)]
        if not indices:
            raise ValueError("Input sentence produced no valid tokens.")

        # 3. Prepare tensors
        seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # batch=1
        lengths = torch.tensor([len(indices)])
        model.eval()
        with torch.no_grad():
            embedding = model.encode(seq, lengths)
            return embedding.squeeze(0).cpu().tolist()
    except Exception as e:
        logger.error(f"[encode_embedding_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[encode_embedding_usecase] {str(e)}")
