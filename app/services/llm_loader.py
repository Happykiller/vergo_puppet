# app\services\llm_loader.py
import os
import time
import psutil
from llama_cpp import Llama

from app.services.logger import logger
from app.apis.common import format_duration

MODEL_PATH = "models/devstralQ4_K_M.gguf"
MAX_CONTEXT_TOKENS = 32768
_llm = None
verbose = False

def get_optimal_thread_count() -> int:
    return min(12, psutil.cpu_count(logical=False) or os.cpu_count() or 8)

def get_llm() -> Llama:
    """Lazy-load the GGUF LLaMA model and return the singleton instance."""
    global _llm

    if _llm is not None:
        logger.info("LLaMA model already loaded and cached.")
        return _llm

    n_threads = get_optimal_thread_count()
    n_gpu_layers = 20

    logger.info("🧠 Preparing to load LLaMA model with the following parameters:")
    logger.info(f"   • Model path: {MODEL_PATH}")
    logger.info(f"   • Model supports ctx up to 131072 tokens — using {MAX_CONTEXT_TOKENS} for now")
    logger.info(f"   • Threads: {n_threads}")
    logger.info(f"   • GPU layers: {n_gpu_layers}")
    logger.info(f"   • Verbose: {verbose}")
    
    start = time.perf_counter()

    _llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=MAX_CONTEXT_TOKENS,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose, 
        jinja=True
    )

    elapsed = time.perf_counter() - start
    logger.info(f"✅ LLaMA model ready in {format_duration(elapsed)} (ctx={MAX_CONTEXT_TOKENS}).")
    return _llm
