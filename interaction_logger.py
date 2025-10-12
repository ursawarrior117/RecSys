# interaction_logger.py
import pandas as pd
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from db_connection import get_engine, init_db

# Initialize DB on import (ensures table exists)
init_db()

engine = get_engine()

def log_user_interaction(user_id, item_id, interaction):
    """
    Log a new user-item interaction to the database.
    - user_id: string (e.g., "user_1")
    - item_id: integer (e.g., 42)
    - interaction: integer (e.g., 1 = click, 2 = purchase, etc.)
    """
    try:
        df = pd.DataFrame([{
            "user_id": user_id,
            "item_id": item_id,
            "interaction": interaction,
            "timestamp": datetime.now()
        }])
        df.to_sql("interactions", con=engine, if_exists="append", index=False)
        print(f"✅ Logged interaction → user:{user_id} | item:{item_id} | type:{interaction}")
    except SQLAlchemyError as e:
        print(f"❌ Failed to log interaction: {e}")

def load_interactions():
    """
    Load all interactions from the database as a pandas DataFrame.
    Returns DataFrame columns: [id, user_id, item_id, interaction, timestamp]
    """
    try:
        df = pd.read_sql("SELECT * FROM interactions", con=engine)
        print(f"📦 Loaded {len(df)} interactions from DB.")
        return df
    except SQLAlchemyError as e:
        print(f"❌ Failed to load interactions: {e}")
        return pd.DataFrame()
