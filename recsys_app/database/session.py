"""Database session and engine configuration with sample data."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np

from recsys_app.core.config import get_settings
from recsys_app.database.models import Base, User, NutritionItem, FitnessItem
from recsys_app.core.utils import calculate_tdee
import os

settings = get_settings()

# Create database engine
# If using SQLite, allow cross-thread usage and use a file-backed DB by default.
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_sample_data():
    """Initialize the database with sample data."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    # Run lightweight migrations for development (add new columns if missing)
    try:
        _run_dev_migrations()
    except Exception:
        pass
    
    db = SessionLocal()
    try:
        # Add sample users if none exist
        if db.query(User).count() == 0:
            sample_users = [
                User(age=25, weight=70.0, height=175.0, gender="M", activity_level="high", health_goals="MG", sleep_good=1),
                User(age=30, weight=65.0, height=165.0, gender="F", activity_level="medium", health_goals="WL", sleep_good=0),
                User(age=40, weight=80.0, height=180.0, gender="M", activity_level="low", health_goals="WL", sleep_good=1),
                User(age=22, weight=55.0, height=160.0, gender="F", activity_level="high", health_goals="MG", sleep_good=0),
                User(age=35, weight=90.0, height=185.0, gender="M", activity_level="medium", health_goals="WL", sleep_good=1),
                User(age=28, weight=60.0, height=170.0, gender="F", activity_level="low", health_goals="MG", sleep_good=1)
            ]
            db.add_all(sample_users)
            db.commit()
            for u in db.query(User).all():
                if u.tdee is None:
                    u.tdee = calculate_tdee(u.weight, u.height, u.age, u.gender, u.activity_level)
            db.commit()

        # Always clear and reload nutrition items from CSV
        import pandas as pd
        import os
        csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '2021-2023 FNDDS At A Glance - FNDDS Nutrient Values.csv')
        try:
            db.query(NutritionItem).delete()
            db.commit()
            df = pd.read_csv(csv_path, skiprows=1)
            # print(f"[CSV Load] Nutrition CSV columns: {list(df.columns)}")
            # print(f"[CSV Load] Nutrition CSV columns: {list(df.columns)}")
            # Map columns and select only needed columns
            rename_map = {
                "Main food description": "food",
                "Energy (kcal)": "calories",
                "Protein (g)": "protein",
                "Carbohydrate (g)": "carbohydrates",
                "Fiber, total dietary (g)": "fiber",
                "Total Fat (g)": "fat",
                "Magnesium (mg)": "magnesium"
            }
            df = df.rename(columns=rename_map)
            needed_cols = list(rename_map.values())
            selected_cols = [col for col in needed_cols if col in df.columns]
            df = df[selected_cols]
            # Coerce numerics and fill missing
            for col in df.columns:
                if col != "food":
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df["food"] = df["food"].fillna("")
            df = df.dropna(subset=["food"])
            items = []
            seen_foods = set()
            # keep any valid rows (calories>0 and protein>0). Previously this was
            # limited to the top 100 rows; remove that limit so the full CSV is
            # imported when initializing the DB.
            valid_df = df[(df["calories"] > 0) & (df["protein"] > 0)].copy()
            for _, row in valid_df.iterrows():
                food_name = str(row.get("food", "")).strip()
                if not food_name or food_name in seen_foods:
                    continue
                try:
                    item = NutritionItem(
                        food=food_name,
                        calories=float(row.get("calories", 0)),
                        protein=float(row.get("protein", 0)),
                        fat=float(row.get("fat", 0)),
                        carbohydrates=float(row.get("carbohydrates", 0)),
                        fiber=float(row.get("fiber", 0)),
                        sugars=float(row.get("sugars", 0)),
                        cholesterol=float(row.get("cholesterol", 0)),
                        sodium=float(row.get("sodium", 0)),
                        magnesium=float(row.get("magnesium", 0)),
                        caffeine=float(row.get("caffeine", 0))
                    )
                    items.append(item)
                    seen_foods.add(food_name)
                except Exception:
                    continue
            db.add_all(items)
            db.commit()
        except Exception as e:
            print(f"[CSV Load] Failed to load nutrition items from CSV: {e}")

        # Always clear and reload fitness items from CSV
        fitness_csv_path = os.path.join(os.path.dirname(__file__), '../../data/megaGymDataset.csv')
        fitness_csv_path = os.path.abspath(fitness_csv_path)
        try:
            db.query(FitnessItem).delete()
            db.commit()
            fdf = pd.read_csv(fitness_csv_path)
            # Map columns and select only needed columns
            fitness_rename_map = {
                "Title": "name",
                "Type": "type",
                "Level": "level",
                "Equipment": "equipment",
                "BodyPart": "bodypart"
            }
            fdf = fdf.rename(columns=fitness_rename_map)
            needed_fit_cols = list(fitness_rename_map.values())
            selected_fit_cols = [col for col in needed_fit_cols if col in fdf.columns]
            fdf = fdf[selected_fit_cols]
            # Fill missing columns and drop rows missing name
            for col in needed_fit_cols:
                if col not in fdf.columns:
                    fdf[col] = ""
            fdf = fdf.fillna({c: "" for c in needed_fit_cols})
            fdf = fdf.dropna(subset=["name"])
            items = []
            seen_names = set()
            valid_fdf = fdf[fdf["name"].notnull() & (fdf["name"] != "")].copy()
            for _, row in valid_fdf.iterrows():
                name = str(row.get("name", "")).strip()
                if not name or name in seen_names:
                    continue
                try:
                    item = FitnessItem(
                        name=name,
                        type=str(row.get("type", "")),
                        level=str(row.get("level", "")),
                        equipment=str(row.get("equipment", "")),
                        bodypart=str(row.get("bodypart", ""))
                    )
                    items.append(item)
                    seen_names.add(name)
                except Exception:
                    continue
            db.add_all(items)
            db.commit()
        except Exception as e:
            print(f"[CSV Load] Failed to load fitness items from megaGymDataset.csv: {e}")
        # Add sample interactions for diversity if none exist
        from recsys_app.database.models import Interaction
        if db.query(Interaction).count() == 0:
            users = db.query(User).all()
            nutrition_items = db.query(NutritionItem).all()
            fitness_items = db.query(FitnessItem).all()
            interactions = []
            # Each user interacts with 2 nutrition and 2 fitness items
            for i, user in enumerate(users):
                nut_ids = [nutrition_items[j % len(nutrition_items)].id for j in range(i, i+2)]
                fit_ids = [fitness_items[j % len(fitness_items)].id for j in range(i, i+2)]
                for nid in nut_ids:
                    interactions.append(Interaction(user_id=user.id, nutrition_item_id=nid, rating=1))
                for fid in fit_ids:
                    interactions.append(Interaction(user_id=user.id, fitness_item_id=fid, rating=1))
            db.add_all(interactions)
            db.commit()

    finally:
        db.close()


def _run_dev_migrations():
    """Run simple ALTER TABLE migrations for development SQLite DB.

    This is intentionally conservative: it only adds the `event_type` column
    to `interactions` if it does not already exist. For production, use a
    proper migration tool (Alembic).
    """
    # Only apply for SQLite file DBs; guard against other DBs for safety
    try:
        if not settings.DATABASE_URL.startswith('sqlite'):
            return
        conn = engine.connect()
        try:
            # Try to add the column unconditionally; ignore errors if it already exists.
            try:
                # Use exec_driver_sql for raw SQL execution on SQLAlchemy 1.4+/2.x
                conn.exec_driver_sql("ALTER TABLE interactions ADD COLUMN event_type VARCHAR")
                print('[MIGRATION] Added event_type column to interactions')
            except Exception as e:
                # Column may already exist or another issue occurred; print debug and continue
                print(f'[MIGRATION] ALTER TABLE skipped or failed (OK to ignore): {e}')
            # Also ensure new user columns added in recent schema updates
            try:
                conn.exec_driver_sql("ALTER TABLE users ADD COLUMN name VARCHAR")
                print('[MIGRATION] Added name column to users')
            except Exception as e:
                print(f'[MIGRATION] ALTER TABLE users.name skipped or failed (OK to ignore): {e}')
            try:
                conn.exec_driver_sql("ALTER TABLE users ADD COLUMN external_id VARCHAR")
                print('[MIGRATION] Added external_id column to users')
            except Exception as e:
                print(f'[MIGRATION] ALTER TABLE users.external_id skipped or failed (OK to ignore): {e}')
        finally:
            conn.close()
    except Exception:
        pass

def get_db():
    """Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()