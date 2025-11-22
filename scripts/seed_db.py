"""Script for seeding the database with sample data."""
import os
import sys
from pathlib import Path
import pandas as pd

# Add the project root directory to the Python path
# scripts/seed_db.py is in <project_root>/scripts/, so parent.parent is the project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from recsys_app.database.session import init_sample_data


def seed_db():
    """Ensure the DB tables exist and populate minimal sample data.

    For the in-memory default DB this must be run in the same process
    so tables are created and basic sample rows are available for testing.
    """
    init_sample_data()
    print("Sample data initialized (users, nutrition items, fitness items).")

if __name__ == "__main__":
    seed_db()