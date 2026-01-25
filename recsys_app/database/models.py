"""SQLAlchemy models for the RecSys application."""
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table, DateTime
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model for storing user profile data."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    # Human-readable name for tracking progress (optional for older rows)
    name = Column(String, nullable=True)
    # External identifier (string) useful for analytics or client-side tracking
    external_id = Column(String, nullable=True)
    age = Column(Integer)
    weight = Column(Float)
    height = Column(Float)
    gender = Column(String)
    activity_level = Column(String)
    health_goals = Column(String)
    sleep_good = Column(Integer)
    tdee = Column(Float)
    # Dietary restrictions: comma-separated string (e.g., "vegetarian,gluten-free,dairy-free")
    dietary_restrictions = Column(String, nullable=True, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NutritionItem(Base):
    """Nutrition item model for storing food data."""
    __tablename__ = "nutrition_items"
    
    id = Column(Integer, primary_key=True)
    food = Column(String, unique=True)
    calories = Column(Float)
    protein = Column(Float)
    fat = Column(Float)
    carbohydrates = Column(Float)
    fiber = Column(Float)
    sugars = Column(Float)
    cholesterol = Column(Float)
    sodium = Column(Float)
    magnesium = Column(Float)
    caffeine = Column(Float)
    # Dietary tags: comma-separated (e.g., "vegetarian,gluten-free,high-protein")
    dietary_tags = Column(String, nullable=True, default="")
    # Meal context: which meal(s) it suits ("breakfast", "lunch", "dinner", or comma-separated)
    meal_context = Column(String, nullable=True, default="")

class FitnessItem(Base):
    """Fitness item model for storing exercise data."""
    __tablename__ = "fitness_items"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    level = Column(String)
    bodypart = Column(String)
    equipment = Column(String)
    type = Column(String)

class Interaction(Base):
    """Interaction model for storing user interactions."""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    nutrition_item_id = Column(Integer, ForeignKey('nutrition_items.id'), nullable=True)
    fitness_item_id = Column(Integer, ForeignKey('fitness_items.id'), nullable=True)
    rating = Column(Float)
    event_type = Column(String, default='impression')
    timestamp = Column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    """Record of a training run and its metrics."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True)
    model_type = Column(String)  # e.g., 'nutrition' or 'fitness' or 'both'
    version = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, default='running')
    metrics = Column(String, nullable=True)  # JSON-encoded metrics or brief summary