"""Recommendation-related API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
from math import exp

from ..database.session import get_db
from ..database.models import User, NutritionItem, FitnessItem, Interaction
from ..schemas.schemas import RecommendationResponse
from ..models.nutrition.utils import hybrid_nutrition_recommendations
from recsys_app.core.utils import calculate_tdee
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

    # Compute BMR/TDEE and nutrition targets once so scorer can use them
    try:
        w = float(user.weight)
    except Exception:
        w = 70.0
    try:
        h = float(user.height)
    except Exception:
        h = 170.0
    try:
        a = float(user.age)
    except Exception:
        a = 30.0
    try:
        gender = (user.gender or '').strip().upper()
    except Exception:
        gender = 'M'
    try:
        # Accept 'M', 'MALE', 'male' etc. Treat first char 'M' as male
        if gender and gender[0] == 'M':
            bmr = 10 * w + 6.25 * h - 5 * a + 5
        else:
            bmr = 10 * w + 6.25 * h - 5 * a - 161
    except Exception:
        bmr = None
    try:
        # Prefer stored user.tdee if present, otherwise calculate from attributes
        if getattr(user, 'tdee', None):
            tdee = float(user.tdee)
        else:
            tdee = calculate_tdee(w, h, a, user.gender, user.activity_level)
    except Exception:
        tdee = None
    try:
        if (user.health_goals or '').upper().startswith('MG'):
            protein_goal = w * 1.6
        else:
            protein_goal = w * 1.2
    except Exception:
        protein_goal = w * 1.2

    protein_cal = protein_goal * 4.0
    remaining_cal = max(0.0, (tdee or (bmr or 2000)) - protein_cal)
    carbs_cal = remaining_cal * 0.6
    fat_cal = remaining_cal * 0.4
    carbs_g = carbs_cal / 4.0
    fat_g = fat_cal / 9.0

    nutrition_targets = {
        'calories': float(tdee) if tdee is not None else None,
        'protein_g': float(protein_goal),
        'carbs_g': float(carbs_g),
        'fat_g': float(fat_g),
        'bmr': float(bmr) if bmr is not None else None
    }
    
    # Determine adaptive alpha (collaborative vs content) based on user's interaction history
    try:
        interaction_count = db.query(Interaction).filter(Interaction.user_id == user.id).count()
    except Exception:
        interaction_count = 0
    # simple rule: cold users (<3 interactions) -> content heavy; active users -> collaborative heavy
    if interaction_count < 3:
        alpha = 0.2
    elif interaction_count < 10:
        alpha = 0.5
    else:
        alpha = 0.8

    # Generate recommendations. Import heavy ML recommenders lazily so the
    # application can start without TensorFlow if it's not available.
    nutrition_recs = None
    fitness_recs = None
    nutr_rec = None  # Initialize to None; may be set below if ML is enabled
    if not DISABLE_ML:
        try:
            from ..models.nutrition.recommender import NutritionRecommender
            from ..models.fitness.recommender import FitnessRecommender
            nutr_rec = NutritionRecommender()
            fit_rec = FitnessRecommender()
            # choose stronger content weights when nutrition targets are available
            prot_w = 0.15
            size_w = 0.35
            if nutrition_targets and nutrition_targets.get('calories'):
                prot_w = 0.6
                size_w = 0.5
            nutrition_recs = hybrid_nutrition_recommendations(
                user_df.iloc[0].to_dict(),
                nutrition_df,
                nutr_rec,
                top_k=top_k,
                alpha=alpha,
                nutrition_targets=nutrition_targets,
                prot_boost_weight=prot_w,
                size_penalty_weight=size_w,
                meals_per_day=3
            )
            fitness_recs = fit_rec.generate_recommendations(user_df.iloc[0].to_dict(), top_k=top_k)
        except Exception:
            # Fall back to content-only scoring if models or TF aren't available
            try:
                prot_w = 0.30
                size_w = 0.35
                if nutrition_targets and nutrition_targets.get('calories'):
                    prot_w = 0.6
                    size_w = 0.5
                nutrition_recs = hybrid_nutrition_recommendations(
                    user_df.iloc[0].to_dict(), nutrition_df, None, top_k=top_k,
                    alpha=alpha,
                    nutrition_targets=nutrition_targets,
                    prot_boost_weight=prot_w,
                    size_penalty_weight=size_w,
                    meals_per_day=3
                )
            except Exception:
                nutrition_recs = nutrition_df.sort_values('protein', ascending=False).head(top_k)
            fitness_recs = fitness_df.head(top_k)
    else:
        # ML disabled by environment variable — use content-only fallback
        try:
            prot_w = 0.30
            size_w = 0.35
            if nutrition_targets and nutrition_targets.get('calories'):
                prot_w = 0.6
                size_w = 0.5
            nutrition_recs = hybrid_nutrition_recommendations(
                user_df.iloc[0].to_dict(), nutrition_df, None, top_k=top_k,
                alpha=alpha,
                nutrition_targets=nutrition_targets,
                prot_boost_weight=prot_w,
                size_penalty_weight=size_w,
                meals_per_day=3
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
            'fiber', 'sugars', 'cholesterol', 'sodium', 'magnesium', 'caffeine',
            'hybrid_score'
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


    def _normalize_fit(row):
        keys = ['id', 'name', 'type', 'level', 'equipment', 'bodypart']
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

    # fitness_recs may be a DataFrame or list of dicts
    try:
        if isinstance(fitness_recs, _pd.DataFrame):
            for _, r in fitness_recs.iterrows():
                fit_list.append(_normalize_fit(r.to_dict()))
        else:
            for r in (fitness_recs if fitness_recs is not None else []):
                fit_list.append(_normalize_fit(r if isinstance(r, dict) else dict(r)))
    except Exception:
        pass

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

    # Build a meal plan using nutrition utilities if nutrition_targets are present
    meal_plan_output = None
    try:
        from ..models.nutrition.utils import generate_daily_meal_plan
        meal_plan = generate_daily_meal_plan(
            user_df.iloc[0].to_dict(),
            nutrition_df,
            nutr_rec,                  # or None
            meals=['Breakfast', 'Lunch', 'Dinner'],
            items_per_meal=2,
            top_k_candidates=80,
            nutrition_targets=nutrition_targets,
            meals_per_day=3,
            diversity_overlap_threshold=0.5
        )
        # normalize meal_plan rows to same schema as nutrition items
        meal_plan_output = {}
        for m, items in meal_plan.items():
            out_items = []
            for it in items:
                out_items.append({
                    'id': it.get('id'),
                    'food': it.get('food'),
                    'calories': it.get('calories'),
                    'protein': it.get('protein'),
                    'fat': it.get('fat'),
                    'carbohydrates': it.get('carbohydrates')
                })
            meal_plan_output[m] = out_items
    except Exception as e:
        print(f'[Recommendation] Meal plan generation failed: {e}')
        meal_plan_output = None

    # nutrition_goal is protein goal in grams
    nutrition_goal = protein_goal

    # fetch interactions for user in last N days and compute interaction boosts
    cutoff = datetime.utcnow() - timedelta(days=30)
    interactions = db.query(Interaction).filter(
        Interaction.user_id == user_id,
        Interaction.timestamp >= cutoff
    ).all()

    # map event types to base weights and decay parameters
    EVENT_WEIGHTS = {'accept': 1.0, 'click': 0.6, 'impression': 0.05, 'skip': -0.4}
    HALF_LIFE_DAYS = 7.0

    # build item boost dict (item_id -> cumulative boost)
    item_boost = {}
    for it in interactions:
        try:
            age_days = (datetime.utcnow() - it.timestamp).total_seconds() / 86400.0
            decay = 2 ** (-age_days / HALF_LIFE_DAYS)
            w = EVENT_WEIGHTS.get(getattr(it, 'event_type', getattr(it, 'type', None)) or 'impression', 0.0) * decay
            item_id = getattr(it, 'nutrition_item_id', None) or getattr(it, 'fitness_item_id', None)
            if item_id is None:
                continue
            item_boost[item_id] = item_boost.get(item_id, 0.0) + w
        except Exception:
            continue

    # Apply interaction boosts to the nutrition and fitness lists by re-ranking.
    # Use any model/content score present (e.g., 'hybrid_score') as the base so
    # interactions can add or subtract from a meaningful continuous score.
    LAMBDA_INTERACTION = 4.0

    def _safe_float(v, default=None):
        try:
            return float(v)
        except Exception:
            return default

    def _rerank_list(item_list, id_key='id'):
        # attach base score (prefer explicit 'hybrid_score' if present)
        ranked = []
        for idx, item in enumerate(item_list):
            iid = item.get(id_key)
            boost = item_boost.get(iid, 0.0)
            base_score = _safe_float(item.get('hybrid_score'), None)
            if base_score is None:
                # fallback to inverse index so earlier items keep precedence
                base_score = -float(idx)
            final_score = base_score + LAMBDA_INTERACTION * float(boost)
            ranked.append((final_score, item))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in ranked]

    try:
        nutr_list = _rerank_list(nutr_list, id_key='id')
    except Exception:
        pass
    try:
        fit_list = _rerank_list(fit_list, id_key='id')
    except Exception:
        pass
    # compute daily progress summary from meal_plan_output if present
    daily_protein = None
    daily_calories = None
    percent_of_goal = None
    try:
        if meal_plan_output:
            dp = 0.0
            dc = 0.0
            for m, items in meal_plan_output.items():
                for it in items:
                    try:
                        dp += float(it.get('protein') or 0.0)
                    except Exception:
                        pass
                    try:
                        dc += float(it.get('calories') or 0.0)
                    except Exception:
                        pass
            daily_protein = float(dp)
            daily_calories = float(dc)
            try:
                if nutrition_targets and nutrition_targets.get('protein_g'):
                    percent_of_goal = float(min(100.0, (daily_protein / float(nutrition_targets.get('protein_g'))) * 100.0))
                else:
                    percent_of_goal = None
            except Exception:
                percent_of_goal = None
    except Exception:
        daily_protein = None
        daily_calories = None
        percent_of_goal = None

    return RecommendationResponse(
        nutrition_items=nutr_list,
        fitness_items=fit_list,
        tdee=float(tdee) if tdee is not None else None,
        nutrition_goal=float(nutrition_goal) if nutrition_goal is not None else None,
        bmr=float(bmr) if bmr is not None else None,
        nutrition_targets=nutrition_targets,
        meal_plan=meal_plan_output,
        daily_protein=daily_protein,
        daily_calories=daily_calories,
        percent_of_goal=percent_of_goal
    )