"""Simple items API used by the React demo frontend.

Exposes GET /api/items and POST /api/items. The POST will create a
NutritionItem record with default numeric values when possible.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List

from ..database.session import get_db
from ..database.models import NutritionItem, FitnessItem, User
from sqlalchemy.exc import IntegrityError
from ..schemas.schemas import NutritionItemResponse, FitnessItemResponse
from ..recommenders.nutrition.utils import hybrid_nutrition_recommendations

router = APIRouter()


@router.get("/items", response_model=List[NutritionItemResponse])
def list_items(db: Session = Depends(get_db)):
    """Return nutrition items with detail fields for the frontend."""
    items = db.query(NutritionItem).all()
    return items


@router.post("/items", response_model=NutritionItemResponse)
def create_item(payload: dict, db: Session = Depends(get_db)):
    """Create a nutrition item (simple demo creation)."""
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name' in body")

    # Ensure numeric defaults
    calories = float(payload.get("calories", 0.0) or 0.0)
    protein = float(payload.get("protein", 0.0) or 0.0)
    fat = float(payload.get("fat", 0.0) or 0.0)
    carbohydrates = float(payload.get("carbohydrates", 0.0) or 0.0)

    # Try to insert; on unique constraint failure, retry with a suffix to ensure
    # frontend add-item works repeatedly.
    base_name = name
    suffix = 0
    while True:
        try:
            candidate = base_name if suffix == 0 else f"{base_name} {suffix}"
            item = NutritionItem(
                food=candidate,
                calories=calories,
                protein=protein,
                fat=fat,
                carbohydrates=carbohydrates,
            )
            db.add(item)
            db.commit()
            db.refresh(item)
            return item
        except IntegrityError:
            db.rollback()
            suffix += 1
            if suffix > 10:
                # Give up after a few attempts
                raise HTTPException(status_code=409, detail="Could not create unique item name")
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))


# Note: user creation is handled by the `users` router. The duplicate
# helper that previously created users under /api/users was removed to
# avoid routing conflicts — use the canonical `/users` (or `/api/users`
# via the CRA proxy) endpoints instead.


@router.get("/recommendations/{user_id}")
def frontend_recommendations(request: Request, user_id: int, top_k: int = 5, db: Session = Depends(get_db)):
    """Return recommendations formatted for the frontend demo.

    This uses the in-memory sample data and simple recommender classes.
    """
    # load user
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # load items
    nutrition_items = db.query(NutritionItem).all()
    fitness_items = db.query(FitnessItem).all()

    # Convert to DataFrame-like records for simple scoring
    import pandas as pd
    user_dict = {k: v for k, v in user.__dict__.items() if not k.startswith("_")}
    nutrition_df = pd.DataFrame([i.__dict__ for i in nutrition_items])
    fitness_df = pd.DataFrame([f.__dict__ for f in fitness_items])

    # Prefer loaded recommenders from app state (if models were loaded on startup)
    recommenders = getattr(request.app.state, 'recommenders', None)

    # Nutrition recommendations: use loaded model if available
    nutrition_recs = None
    if recommenders and recommenders.get('nutrition') is not None:
        try:
            nutr_rec = recommenders['nutrition']
            nutrition_recs = nutr_rec.generate_recommendations(user_dict, top_k=top_k)
        except Exception:
            nutrition_recs = None

    if nutrition_recs is None:
        # fallback to content-based hybrid util or simple protein sort
        try:
            # Import recommender lazily for content scoring helper only when needed
            from ..models.nutrition.recommender import NutritionRecommender
            nutrition_recs = hybrid_nutrition_recommendations(user_dict, nutrition_df, NutritionRecommender(), top_k=top_k)
        except Exception:
            nutrition_recs = nutrition_df.sort_values("protein", ascending=False).head(top_k)

    # Fitness recommendations: use loaded model if available
    fitness_recs = None
    if recommenders and recommenders.get('fitness') is not None:
        try:
            fit_rec = recommenders['fitness']
            fitness_recs = fit_rec.generate_recommendations(user_dict, top_k=top_k)
        except Exception:
            fitness_recs = None

    if fitness_recs is None:
        # Fallback: simple level match
        try:
            fitness_recs = fitness_df[fitness_df["level"] == user.activity_level].head(top_k)
        except Exception:
            fitness_recs = fitness_df.head(top_k)

    # Sanitize output: replace NaN/None with safe defaults
    def sanitize_records(records):
        sanitized = []
        for rec in records:
            clean = {}
            for k, v in rec.items():
                if v is None:
                    clean[k] = 0 if isinstance(v, (int, float)) else ""
                elif isinstance(v, float) and (v != v or v is None):  # NaN check
                    clean[k] = 0
                else:
                    clean[k] = v
            sanitized.append(clean)
        return sanitized

    nutrition_list = nutrition_recs.to_dict(orient="records")
    fitness_list = fitness_recs.to_dict(orient="records")
    return {
        "nutrition_items": sanitize_records(nutrition_list),
        "fitness_items": sanitize_records(fitness_list),
    }
