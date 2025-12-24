"""FastAPI application creation and configuration."""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core.config import get_settings
from .database.session import engine, init_sample_data
from .database.models import Base
from apscheduler.schedulers.background import BackgroundScheduler
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI application."""
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    import os
    # Suppress data loading prints
    with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
        # Create database tables and initialize sample data
        Base.metadata.create_all(bind=engine)
        init_sample_data()
        # For stability on systems without a TF-compatible runtime, do not import
        # the training/model code at startup. Models can be trained/loaded via the
        # admin endpoints which import training lazily. Keep initial state empty.
        app.state.recommenders = None
        app.state.scheduler = None

        # Automate model retraining on startup unless ML is explicitly disabled
        if os.getenv('RECSYS_DISABLE_ML', '0') != '1':
            import threading
            def retrain_models():
                try:
                    from recsys_app.services.training import train_and_persist_models
                    train_and_persist_models(simulate_interactions=True)
                except Exception as e:
                    print(f"[Startup Retrain] Model training failed: {e}")
            threading.Thread(target=retrain_models, daemon=True).start()

        # Schedule a daily check that retrains models when enough new interactions are present
        try:
            from recsys_app.services.training import retrain_if_needed
            sched = BackgroundScheduler()
            def _scheduled_check():
                try:
                    res = retrain_if_needed(min_new_interactions=50, window_days=7)
                    print(f"[Scheduled Retrain Check] {res}")
                except Exception as e:
                    print(f"[Scheduled Retrain] failed: {e}")
            # Run once a day at 02:00 UTC (change as needed) — here we use an interval of 24 hours
            sched.add_job(_scheduled_check, 'interval', hours=24, id='daily_retrain_check')
            sched.start()
            app.state.scheduler = sched
        except Exception:
            # If scheduler setup fails, continue without scheduled retraining
            print("[Scheduler] failed to start scheduled retrain job")

    yield
    # Shutdown scheduler on app shutdown
    sched = getattr(app.state, 'scheduler', None)
    if sched:
        try:
            sched.shutdown(wait=False)
        except Exception:
            pass

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        lifespan=lifespan
    )
    
    # Include routers
    from .routes import users, recommendations, items, admin, interactions
    # Mount users under /api/users so the CRA dev-server proxy (/api/*) can access them
    # This is the primary mount for the frontend; /users is kept for direct backend access
    app.include_router(users.router, prefix="/api/users", tags=["users"])
    # Also mount at /users for non-proxied direct access
    app.include_router(users.router, prefix="/users", tags=["users_direct"])
    # Expose recommendations under /api so the CRA frontend proxy ("/api/*") works
    app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
    # Add the items router under the /api prefix so CRA proxy can forward /api requests
    app.include_router(items.router, prefix="/api", tags=["api"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])
    app.include_router(interactions.router, prefix="/api", tags=["api"])
    
    return app