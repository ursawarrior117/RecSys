"""Endpoints to record user interactions for training data."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from ..database.session import get_db
from ..database.models import Interaction, User, NutritionItem, FitnessItem

router = APIRouter()


@router.post("/interactions")
def log_interaction(payload: dict, db: Session = Depends(get_db)):
    """Log a user interaction. payload should include: user_id, nutrition_item_id (optional), fitness_item_id (optional), rating (float)."""
    user_id = payload.get('user_id')
    if user_id is None:
        raise HTTPException(status_code=400, detail='user_id is required')
    # validate user
    if not db.query(User).filter(User.id == user_id).first():
        raise HTTPException(status_code=404, detail='user not found')

    ni = payload.get('nutrition_item_id')
    fi = payload.get('fitness_item_id')
    rating = float(payload.get('rating', 1.0))

    if ni is None and fi is None:
        raise HTTPException(status_code=400, detail='nutrition_item_id or fitness_item_id required')

    if ni is not None and not db.query(NutritionItem).filter(NutritionItem.id == ni).first():
        raise HTTPException(status_code=404, detail='nutrition item not found')
    if fi is not None and not db.query(FitnessItem).filter(FitnessItem.id == fi).first():
        raise HTTPException(status_code=404, detail='fitness item not found')

    inter = Interaction(user_id=user_id, nutrition_item_id=ni, fitness_item_id=fi, rating=rating)
    db.add(inter)
    db.commit()
    db.refresh(inter)
    return { 'id': inter.id, 'status': 'recorded' }
