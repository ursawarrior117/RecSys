"""Recommendation-related API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database.session import get_db
from ..database.models import User, NutritionItem, FitnessItem
from ..schemas.schemas import RecommendationResponse
from ..models.nutrition.utils import hybrid_nutrition_recommendations
import os

# If set, skip importing ML/TensorFlow-based recommenders and use content-only
# fallbacks. Useful for running the demo on machines without a TF-compatible
# runtime or to avoid heavy native imports.
DISABLE_ML = os.getenv('RECSYS_DISABLE_ML', '0') == '1'

router = APIRouter()

@router.get("/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    db: Session = Depends(get_db),
    top_k: int = 5
):
    """Get personalized recommendations for a user."""
    # Get user
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get items from database
    nutrition_items = db.query(NutritionItem).all()
    fitness_items = db.query(FitnessItem).all()
    
    # Convert to pandas DataFrames
    import pandas as pd
    user_df = pd.DataFrame([user.__dict__])
    nutrition_df = pd.DataFrame([item.__dict__ for item in nutrition_items])
    fitness_df = pd.DataFrame([item.__dict__ for item in fitness_items])
    
    # Generate recommendations. Import heavy ML recommenders lazily so the
    # application can start without TensorFlow if it's not available.
    nutrition_recs = None
    fitness_recs = None
    if not DISABLE_ML:
        try:
            from ..models.nutrition.recommender import NutritionRecommender
            from ..models.fitness.recommender import FitnessRecommender
            nutr_rec = NutritionRecommender()
            fit_rec = FitnessRecommender()
            nutrition_recs = hybrid_nutrition_recommendations(
                user_df.iloc[0].to_dict(),
                nutrition_df,
                nutr_rec,
                top_k=top_k
            )
            fitness_recs = fit_rec.generate_recommendations(user_df.iloc[0].to_dict(), top_k=top_k)
        except Exception:
            # Fall back to content-only scoring if models or TF aren't available
            try:
                nutrition_recs = hybrid_nutrition_recommendations(
                    user_df.iloc[0].to_dict(), nutrition_df, None, top_k=top_k
                )
            except Exception:
                nutrition_recs = nutrition_df.sort_values('protein', ascending=False).head(top_k)
            fitness_recs = fitness_df.head(top_k)
    else:
        # ML disabled by environment variable — use content-only fallback
        try:
            nutrition_recs = hybrid_nutrition_recommendations(
                user_df.iloc[0].to_dict(), nutrition_df, None, top_k=top_k
            )
        except Exception:
            nutrition_recs = nutrition_df.sort_values('protein', ascending=False).head(top_k)
        fitness_recs = fitness_df.head(top_k)
    
    # Convert DataFrames to plain dicts matching the response schema
    import math
    import pandas as _pd

    def _normalize_nut(row):
        keys = [
            'id', 'food', 'calories', 'protein', 'fat', 'carbohydrates',
            'fiber', 'sugars', 'cholesterol', 'sodium', 'magnesium', 'caffeine'
        ]
        out = {}
        for k in keys:
            v = row.get(k) if isinstance(row, dict) else row.get(k, None)
            # convert pandas / numpy missing values to None
            try:
                if _pd.isna(v):
                    v = None
            except Exception:
                pass
            # convert numpy scalar types to native Python
            try:
                if v is not None and (hasattr(v, 'item') or hasattr(v, 'dtype')):
                    v = v.item()
            except Exception:
                pass
            # final check: reject NaN floats
            try:
                if isinstance(v, float) and math.isnan(v):
                    v = None
            except Exception:
                pass
            out[k] = v
        return out

    def _normalize_fit(row):
        keys = ['id', 'name', 'level', 'bodypart', 'equipment', 'type']
        out = {}
        for k in keys:
            v = row.get(k) if isinstance(row, dict) else row.get(k, None)
            try:
                if _pd.isna(v):
                    v = None
            except Exception:
                pass
            try:
                if v is not None and (hasattr(v, 'item') or hasattr(v, 'dtype')):
                    v = v.item()
            except Exception:
                pass
            try:
                if isinstance(v, float) and math.isnan(v):
                    v = None
            except Exception:
                pass
            out[k] = v
        return out

    nutr_list = []
    fit_list = []
    # nutrition_recs may be a DataFrame or list of dicts
    try:
        for _, r in nutrition_recs.iterrows():
            nutr_list.append(_normalize_nut(r.to_dict()))
    except Exception:
        # try treating as list
        for r in (nutrition_recs or []):
            nutr_list.append(_normalize_nut(r if isinstance(r, dict) else dict(r)))

    try:
        for _, r in fitness_recs.iterrows():
            fit_list.append(_normalize_fit(r.to_dict()))
    except Exception:
        for r in (fitness_recs or []):
            fit_list.append(_normalize_fit(r if isinstance(r, dict) else dict(r)))

    # Final sanitization: replace any remaining NaN/Inf with None to ensure
    # JSON serialization succeeds.
    for d in nutr_list:
        for k, v in list(d.items()):
            try:
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    d[k] = None
            except Exception:
                pass
    for d in fit_list:
        for k, v in list(d.items()):
            try:
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    d[k] = None
            except Exception:
                pass

    return RecommendationResponse(nutrition_items=nutr_list, fitness_items=fit_list)