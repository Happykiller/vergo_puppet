# app\services\llm_loader.py
import time
import importlib
from pathlib import Path
from llama_cpp import Llama

from app.services.logger import logger

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
MAX_CONTEXT_TOKENS = 16384
_llm = None

def _get_llama_version_safe() -> str:
    try:
        module = importlib.import_module("llama_cpp")
        return str(getattr(module, "llama_version", lambda: "unknown")())
    except Exception as e:
        logger.warning(f"Unable to retrieve llama_version: {e}")
        return "unknown"

def _check_cuda_support() -> bool:
    """
    Check for compiled CUDA support via the common dynamic libraries used by llama.cpp
    """
    try:
        import llama_cpp
        lib_path = getattr(llama_cpp, "__path__", None)
        if not lib_path:
            logger.debug("llama_cpp has no __path__, cannot inspect shared libs.")
            return False

        # Looking for compiled backends
        lib_dir = Path(lib_path[0])
        files = list(lib_dir.glob("*.so")) + list(lib_dir.glob("*.dylib")) + list(lib_dir.glob("*.dll"))
        for f in files:
            if "cuda" in f.name.lower() or "ggml-cuda" in f.name.lower():
                logger.debug(f"Detected CUDA support via shared object: {f.name}")
                return True

        return False
    except Exception as e:
        logger.warning(f"Could not check CUDA support: {e}")
        return False

def get_llm() -> Llama:
    """Lazy-load the GGUF LLaMA model and return the singleton instance."""
    global _llm

    if _llm is not None:
        logger.info("LLaMA model already loaded and cached.")
        return _llm

    logger.debug(f"LLaMA backend version: {_get_llama_version_safe()}")

    has_cuda = _check_cuda_support()
    if has_cuda:
        logger.info("🚀 CUDA support detected in llama-cpp backend.")
    else:
        logger.warning("⚠️ CUDA support **NOT** detected in llama-cpp. Running in CPU-only mode.")

    logger.info("🧠 Loading LLaMA model from GGUF...")
    start = time.perf_counter()

    gpu_layers = 12 if has_cuda else 0

    _llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=MAX_CONTEXT_TOKENS,
        n_threads=8,
        n_gpu_layers=gpu_layers,
        verbose=True
    )

    elapsed = time.perf_counter() - start
    logger.info(f"✅ LLaMA model ready in {elapsed:.2f} seconds (ctx={MAX_CONTEXT_TOKENS}).")
    return _llm
