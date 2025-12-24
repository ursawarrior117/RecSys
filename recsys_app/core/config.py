"""Configuration settings for the RecSys application."""
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    APP_NAME: str = "RecSys"
    DEBUG: bool = True
    # Use file-backed SQLite by default so the server processes share the same DB.
    DATABASE_URL: str = "sqlite:///./recsys.db"
    MODEL_PATH: str = "saved_models"
    
    # Allow extra environment variables (so .env can contain postgres_* vars etc.)
    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()