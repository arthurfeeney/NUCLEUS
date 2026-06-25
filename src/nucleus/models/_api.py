import dataclasses
from typing import Optional, Callable, TypeVar, List, Any
import torch
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)
MODELS = {}

def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name or fn.__name__
        if key in MODELS:
            raise ValueError(f"Cannot register duplicate model ({key})")
        MODELS[key] = fn
        return fn
    return wrapper

def list_models() -> List[str]:
    return sorted(list(MODELS.keys()), key=lambda x: x[0])

def get_model(name: str, **kwargs: Any) -> nn.Module:
    name = name.lower()
    try:
        fn = MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Model {name} not found. Available Models: {MODELS.keys()}") from exc

    if hasattr(fn, "config_class"):
        valid_fields = {f.name for f in dataclasses.fields(fn.config_class)}
        config = fn.config_class(**{k: v for k, v in kwargs.items() if k in valid_fields})
        model = fn(config)
        model._model_name = name
        return model
    return fn(**kwargs)

def get_model_class(name: str) -> nn.Module:
    name = name.lower()
    try:
        fn = MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Model {name} not found. Available Models: {MODELS.keys()}") from exc
    return fn

def load_model_from_checkpoint(path, map_location=None) -> nn.Module:
    """
    Reconstruct a model and load its weights from a checkpoint, using weights_only=True.

    Supports both Lightning checkpoints (state dict nested under "state_dict" with a
    "model." prefix on every key) and raw torch.save(model.state_dict(), path) saves.

    Usage:
        model = load_model_from_checkpoint("run/checkpoints/last.ckpt")
        model = load_model_from_checkpoint("last.ckpt", map_location="cuda")
    """
    try:
        ckpt = torch.load(path, weights_only=True, map_location=map_location)
    except:
        ckpt = torch.load(path, weights_only=False, map_location=map_location)

    if "state_dict" in ckpt:
        # Lightning checkpoint — strip the "model." prefix
        model_state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    else:
        model_state = ckpt

    extra = model_state.get("_extra_state")
    if extra is None:
        raise ValueError("No '_extra_state' in checkpoint. Was this model saved with a config dataclass?")

    model_name = extra.get("model_name")
    config_dict = extra.get("config")
    if model_name is None or config_dict is None:
        raise ValueError("'_extra_state' is missing 'model_name' or 'config'.")

    fn = MODELS[model_name]
    if hasattr(fn, "config_from_dict"):
        config = fn.config_from_dict(config_dict)
        model = fn(config)
        model._model_name = model_name
    else:
        model = get_model(model_name, **config_dict)
    model.load_state_dict(model_state)
    return model
