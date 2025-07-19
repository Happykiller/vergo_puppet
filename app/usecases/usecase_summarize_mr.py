# app/usecases/usecase_summarize_mr.py
from llama_cpp import Llama
from pathlib import Path

from app.services.llm_loader import get_llm

def usecase_summarize_mr(prompt_path: str, markdown_path: str) -> str:
    """Summarize MR using llama-cpp-python with GGUF model."""
    prompt_file = Path(prompt_path)
    markdown_file = Path(markdown_path)

    if not prompt_file.exists() or not markdown_file.exists():
        raise FileNotFoundError("Prompt or MR file does not exist")

    prompt = prompt_file.read_text(encoding="utf-8")
    markdown = markdown_file.read_text(encoding="utf-8")
    full_input = f"{prompt.strip()}\n\n---\n\n{markdown.strip()}"

    llm = get_llm()
    response = llm(
        full_input,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>", "## Titre MR", "## Résumé"]
    )

    #response = llm("Quel temps fait-il à Paris aujourd'hui ?", max_tokens=128)

    return response["choices"][0]["text"].strip()