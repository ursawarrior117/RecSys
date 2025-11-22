"""Script for initializing the database."""
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from alembic.config import Config
from alembic import command
from recsys_app.database.session import engine
from recsys_app.database.models import Base

def init_db():
    """Initialize the database with tables and initial migrations."""
    # Create database tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Create alembic version table
    alembic_cfg = Config("alembic.ini")
    command.stamp(alembic_cfg, "head")
    
if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")