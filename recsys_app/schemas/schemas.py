"""Pydantic schemas for request/response models."""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class UserBase(BaseModel):
    age: int
    weight: float
    height: float
    gender: str
    activity_level: str
    health_goals: str
    sleep_good: int

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

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
        from_attributes = True

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
        from_attributes = True

class RecommendationResponse(BaseModel):
    nutrition_items: List[NutritionItemResponse]
    fitness_items: List[FitnessItemResponse]