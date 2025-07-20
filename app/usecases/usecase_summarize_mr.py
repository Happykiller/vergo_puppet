# app/usecases/usecase_summarize_mr.py
from pathlib import Path
from time import perf_counter

from app.services.logger import logger
from app.apis.common import format_duration
from app.services.llm_loader import get_llm

def usecase_summarize_mr(prompt_path: str, markdown_path: str) -> str:
    """Summarize MR using llama-cpp-python with GGUF model."""
    prompt_file = Path(prompt_path)
    markdown_file = Path(markdown_path)

    if not prompt_file.exists() or not markdown_file.exists():
        raise FileNotFoundError("Prompt or MR file does not exist")

    prompt = prompt_file.read_text(encoding="utf-8")
    markdown = markdown_file.read_text(encoding="utf-8")
    system = "[SYSTEM_PROMPT]Tu es Devstral, un assistant code…[/SYSTEM_PROMPT]"
    full_input = system + "\n[INST]" + prompt + markdown + "[/INST]"

    llm = get_llm()
    
    start_time = perf_counter()
    
    response = llm(
        full_input,
        max_tokens=2048,
        temperature=0.1, 
        min_p=0.01, 
        top_p=1.0
    )
    
    duration = perf_counter() - start_time
    
    usage = response.get("usage", {})
    logger.info(
        "[usecase_summarize_mr] ✅ LLM completed | "
        f"Duration: {format_duration(duration)} | "
        f"Prompt tokens: {usage.get('prompt_tokens', '?')} | "
        f"Completion tokens: {usage.get('completion_tokens', '?')} | "
        f"Total tokens: {usage.get('total_tokens', '?')}"
    )

    return response["choices"][0]["text"].strip()