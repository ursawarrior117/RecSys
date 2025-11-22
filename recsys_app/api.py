"""FastAPI application creation and configuration."""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core.config import get_settings
from .database.session import engine, init_sample_data
from .database.models import Base
from apscheduler.schedulers.background import BackgroundScheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI application."""
    # Create database tables and initialize sample data
    Base.metadata.create_all(bind=engine)
    init_sample_data()
    # For stability on systems without a TF-compatible runtime, do not import
    # the training/model code at startup. Models can be trained/loaded via the
    # admin endpoints which import training lazily. Keep initial state empty.
    app.state.recommenders = None
    app.state.scheduler = None
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
    app.include_router(users.router, prefix="/users", tags=["users"])
    # Expose recommendations under /api so the CRA frontend proxy ("/api/*") works
    app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
    # Add the items router under the /api prefix so CRA proxy can forward /api requests
    app.include_router(items.router, prefix="/api", tags=["api"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])
    app.include_router(interactions.router, prefix="/api", tags=["api"])
    
    return app