"""Database session and engine configuration with sample data."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np

from recsys_app.core.config import get_settings
from recsys_app.database.models import Base, User, NutritionItem, FitnessItem
from recsys_app.core.utils import calculate_tdee

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
    
    db = SessionLocal()
    try:
        # Add sample users if none exist
        if db.query(User).count() == 0:
            sample_users = [
                User(
                    age=25,
                    weight=70.0,
                    height=175.0,
                    gender="M",
                    activity_level="high",
                    health_goals="MG",
                    sleep_good=1
                ),
                User(
                    age=30,
                    weight=65.0,
                    height=165.0,
                    gender="F",
                    activity_level="medium",
                    health_goals="WL",
                    sleep_good=0
                )
            ]
            db.add_all(sample_users)
            db.commit()
            # compute and save tdee for seeded users
            for u in db.query(User).all():
                if u.tdee is None:
                    u.tdee = calculate_tdee(u.weight, u.height, u.age, u.gender, u.activity_level)
            db.commit()

        # Add sample nutrition items if none exist
        if db.query(NutritionItem).count() == 0:
            nutrition_items = [
                NutritionItem(
                    food="Chicken Breast",
                    calories=165,
                    protein=31,
                    fat=3.6,
                    carbohydrates=0,
                    fiber=0,
                    magnesium=29
                ),
                NutritionItem(
                    food="Salmon",
                    calories=208,
                    protein=22,
                    fat=13,
                    carbohydrates=0,
                    fiber=0,
                    magnesium=27
                ),
                NutritionItem(
                    food="Quinoa",
                    calories=120,
                    protein=4.4,
                    fat=1.9,
                    carbohydrates=21,
                    fiber=2.8,
                    magnesium=64
                )
            ]
            db.add_all(nutrition_items)
            db.commit()

        # Add sample fitness items if none exist
        if db.query(FitnessItem).count() == 0:
            fitness_items = [
                FitnessItem(
                    name="Push-ups",
                    level="medium",
                    bodypart="chest",
                    equipment="bodyweight",
                    type="strength"
                ),
                FitnessItem(
                    name="Running",
                    level="high",
                    bodypart="legs",
                    equipment="none",
                    type="cardio"
                ),
                FitnessItem(
                    name="Yoga",
                    level="low",
                    bodypart="full_body",
                    equipment="mat",
                    type="flexibility"
                )
            ]
            db.add_all(fitness_items)
            db.commit()

    finally:
        db.close()

def get_db():
    """Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()