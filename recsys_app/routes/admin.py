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


@router.post("/retrain_check")
def retrain_check(min_new_interactions: int = 50, window_days: int = 7, force: bool = False) -> Any:
    """Check recent interactions and retrain models when appropriate.

    - `min_new_interactions`: threshold of recent interactions required to trigger a retrain.
    - `window_days`: look-back window (days) for recent interactions.
    - `force`: if True, force a retrain regardless of counts.
    """
    try:
        if force:
            from ..services.training import train_and_persist_models
            return train_and_persist_models(simulate_interactions=False)
        else:
            from ..services.training import retrain_if_needed
            return retrain_if_needed(min_new_interactions=min_new_interactions, window_days=window_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training_runs")
def list_training_runs(limit: int = 20):
    """List recent training runs."""
    from ..database.session import SessionLocal
    from ..database.models import TrainingRun
    db = SessionLocal()
    try:
        rows = db.query(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(limit).all()
        out = []
        for r in rows:
            out.append({
                'id': r.id,
                'model_type': r.model_type,
                'version': r.version,
                'started_at': r.started_at,
                'finished_at': r.finished_at,
                'status': r.status,
                'metrics': r.metrics,
            })
        return { 'runs': out }
    finally:
        db.close()
