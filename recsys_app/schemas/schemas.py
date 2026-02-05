"""Pydantic schemas for request/response models."""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict

class UserBase(BaseModel):
    age: int
    weight: float
    height: float
    gender: str
    activity_level: str
    health_goals: str
    sleep_good: int
    # Optional name and external id in base for compatibility
    name: Optional[str] = None
    external_id: Optional[str] = None
    # Dietary restrictions: comma-separated string
    dietary_restrictions: Optional[str] = None

class UserCreate(UserBase):
    id: Optional[int] = None  # Optional custom numeric ID provided by user
    # Keep name and external_id optional at schema level so requests
    # that omit them produce a controlled 400 from the route instead
    name: Optional[str] = None
    external_id: Optional[str] = None
    dietary_restrictions: Optional[str] = None

class UserResponse(UserBase):
    id: int
    name: Optional[str] = None
    external_id: Optional[str] = None
    dietary_restrictions: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class NutritionItemBase(BaseModel):
    food: str
    calories: float
    protein: float
    fat: float
    carbohydrates: float
    fiber: Optional[float] = None
    sugars: Optional[float] = None
    cholesterol: Optional[float] = None
    sodium: Optional[float] = None
    magnesium: Optional[float] = None
    caffeine: Optional[float] = None

class NutritionItemCreate(NutritionItemBase):
    pass

class NutritionItemResponse(NutritionItemBase):
    id: int
    
    class Config:
        orm_mode = True

class FitnessItemBase(BaseModel):
    name: str
    level: str
    bodypart: str
    equipment: str
    type: str

class FitnessItemCreate(FitnessItemBase):
    pass

class FitnessItemResponse(FitnessItemBase):
    id: int
    
    class Config:
        orm_mode = True

class RecommendationResponse(BaseModel):
    nutrition_items: List[NutritionItemResponse]
    fitness_items: List[FitnessItemResponse]
    tdee: Optional[float] = None
    bmi: Optional[float] = None
    user_height: Optional[float] = None
    user_weight: Optional[float] = None
    nutrition_goal: Optional[float] = None
    bmr: Optional[float] = None
    nutrition_targets: Optional[dict] = None
    meal_plan: Optional[Dict[str, List[NutritionItemResponse]]] = None
    # Daily progress summary: total protein/calories in the generated meal_plan
    daily_protein: Optional[float] = None
    daily_calories: Optional[float] = None
    daily_fiber: Optional[float] = None
    daily_magnesium: Optional[float] = None
    daily_magnesium_pct: Optional[float] = None
    magnesium_warning: Optional[bool] = None
    percent_of_goal: Optional[float] = None