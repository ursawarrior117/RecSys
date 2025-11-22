"""SQLAlchemy models for the RecSys application."""
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table, DateTime
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model for storing user profile data."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    weight = Column(Float)
    height = Column(Float)
    gender = Column(String)
    activity_level = Column(String)
    health_goals = Column(String)
    sleep_good = Column(Integer)
    tdee = Column(Float)
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
    timestamp = Column(DateTime, default=datetime.utcnow)