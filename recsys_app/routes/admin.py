"""Admin endpoints for maintenance tasks like training models."""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Any

router = APIRouter()


def _run_training_task():
    # Import the training function lazily so we don't import heavy ML libs at startup
    try:
        from ..services.training import train_and_persist_models
    except Exception:
        # If import fails, raise to be handled by the background task
        raise
    # Run synchronously inside background thread/process
    return train_and_persist_models(simulate_interactions=True)


@router.post("/train")
def trigger_training(background_tasks: BackgroundTasks) -> Any:
    """Trigger model training in the background. Returns immediately.

    The actual training will be executed as a background task and models will be
    saved under the configured `MODEL_PATH`.
    """
    background_tasks.add_task(_run_training_task)
    return {"status": "training_started"}


@router.get("/models")
def list_models():
    from ..model_io import list_models as _list
    return {"files": _list()}
