from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from model_wrapper import ModelWrapper

MODEL_PATHS: Dict[str, str] = {
    "qwen3-0.6b": "models/Qwen3-0.6B",
    "qwen3-4b": "models/Qwen3-4B",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_ENABLED = False  # Toggle to True to log to Weights & Biases when credentials are available

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

loaded_models: Dict[str, ModelWrapper] = {}


def get_model_wrapper(
    model_key: str,
    *,
    use_lora: bool = False,
    lora_config: Optional[dict] = None,
    force_reload: bool = False,
) -> ModelWrapper:
    """Return a ModelWrapper for the requested model, loading weights on demand."""
    if model_key not in MODEL_PATHS:
        raise KeyError(f"Unknown model key '{model_key}'. Available: {list(MODEL_PATHS.keys())}")

    wrapper = loaded_models.get(model_key)
    needs_reload = (
        force_reload
        or wrapper is None
        or getattr(wrapper, "model", None) is None
        or (use_lora and not getattr(wrapper, "is_lora", False))
    )

    if wrapper is None:
        wrapper = ModelWrapper(model_name=MODEL_PATHS[model_key], device=DEVICE)
        needs_reload = True

    if needs_reload:
        wrapper.load_model(use_lora=use_lora, lora_config=lora_config)
        loaded_models[model_key] = wrapper

    return wrapper


def release_model(model_key: str) -> None:
    """Remove a loaded model from memory to free GPU/CPU resources."""
    wrapper = loaded_models.pop(model_key, None)
    if wrapper is None:
        return

    model = getattr(wrapper, "model", None)
    try:
        if model is not None:
            try:
                model.to("cpu")
            except Exception as move_exc:
                print(f"Warning: unable to move model for {model_key} to CPU: {move_exc}")
        del model
    except Exception as exc:
        print(f"Warning: cleanup for model {model_key} raised an exception: {exc}")

    wrapper.model = None
    wrapper.tokenizer = None
    wrapper.is_lora = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    try:
        import gc

        gc.collect()
    except Exception:
        pass


def release_all_models() -> None:
    """Release all cached models to ensure GPU memory is freed."""
    for cached_key in list(loaded_models.keys()):
        release_model(cached_key)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    try:
        import gc

        gc.collect()
    except Exception:
        pass


def main() -> None:
    """Lightweight test for the registry helpers."""
    print(f"Available models: {list(MODEL_PATHS.keys())}")
    print(f"Using device: {DEVICE}")
    print("Call get_model_wrapper('<model_key>') to load a model when needed.")


if __name__ == "__main__":
    main()

