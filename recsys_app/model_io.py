"""Helpers to save and load recommender models and preprocessors."""
import os
import joblib
from pathlib import Path
from recsys_app.core.config import get_settings

settings = get_settings()
MODELS_DIR = Path(settings.MODEL_PATH)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_keras_model(model, name: str):
    # Save as a single .keras file (compatible with Keras 3)
    path = MODELS_DIR / f"{name}.keras"
    model.save(str(path))
    return str(path)


def load_keras_model(name: str):
    from tensorflow import keras
    path = MODELS_DIR / f"{name}.keras"
    if not path.exists():
        return None
    return keras.models.load_model(str(path))


def save_preprocessor(obj, name: str):
    path = MODELS_DIR / f"{name}_prep.joblib"
    joblib.dump(obj, str(path))
    return str(path)


def load_preprocessor(name: str):
    path = MODELS_DIR / f"{name}_prep.joblib"
    if not path.exists():
        return None
    return joblib.load(str(path))


def list_models():
    items = []
    for p in MODELS_DIR.iterdir():
        items.append(p.name)
    return items
