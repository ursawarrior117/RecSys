"""Recommendation-related API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
from math import exp

from ..database.session import get_db
from ..database.models import User, NutritionItem, FitnessItem, Interaction
from ..schemas.schemas import RecommendationResponse
from ..recommenders.nutrition.utils import hybrid_nutrition_recommendations
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
    top_k: int = 5,
    rec_type: str = "all",
    bodypart: str = None
):
    """Get personalized recommendations for a user.
    """
    # Validate rec_type
    if rec_type not in ('nutrition', 'fitness', 'all'):
        rec_type = 'all'
    
    print(f'[RECOMMENDATION] User {user_id}, rec_type={rec_type}, bodypart={bodypart}')
    
    # Get user
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get items from database
    nutrition_items = db.query(NutritionItem).all() if rec_type in ('nutrition', 'all') else []
    fitness_items = db.query(FitnessItem).all() if rec_type in ('fitness', 'all') else []
    
    # Filter fitness items by body part if specified
    if bodypart and fitness_items:
        bodypart_lower = bodypart.strip().lower()
        fitness_items = [it for it in fitness_items if bodypart_lower in (getattr(it, 'bodypart', '') or '').lower()]
        print(f'[RECOMMENDATION] Filtered fitness to {len(fitness_items)} items for bodypart={bodypart}')
    
    # **Filter by dietary restrictions**
    user_restrictions = set()
    try:
        if user.dietary_restrictions:
            user_restrictions = set(r.strip().lower() for r in user.dietary_restrictions.split(',') if r.strip())
    except Exception:
        pass
    
    # Filter out items that conflict with user restrictions
    if user_restrictions:
        filtered_items = []
        for it in nutrition_items:
            item_tags = set()
            try:
                if getattr(it, 'dietary_tags', None):
                    item_tags = set(t.strip().lower() for t in it.dietary_tags.split(',') if t.strip())
            except Exception:
                pass
            should_exclude = False
            if "vegetarian" in user_restrictions and "meat" in item_tags:
                should_exclude = True
            if "vegan" in user_restrictions and any(t in item_tags for t in ["meat", "dairy", "eggs"]):
                should_exclude = True
            if "gluten-free" in user_restrictions and "gluten" in item_tags:
                should_exclude = True
            if "dairy-free" in user_restrictions and "dairy" in item_tags:
                should_exclude = True
            if "nut-free" in user_restrictions and "nuts" in item_tags:
                should_exclude = True
            if not should_exclude:
                filtered_items.append(it)
        nutrition_items = filtered_items
    
    # Query user's persistent dislikes (should never appear in recommendations)
    # and likes (should preferentially appear)
    disliked_nutrition_ids = set()
    liked_nutrition_ids = set()
    disliked_fitness_ids = set()
    liked_fitness_ids = set()
    
    try:
        # Nutrition dislikes
        dislike_query = db.query(Interaction).filter(
            Interaction.user_id == user.id,
            Interaction.nutrition_item_id != None,
            Interaction.event_type == 'dislike'
        ).all()
        for it in dislike_query:
            if it.nutrition_item_id:
                disliked_nutrition_ids.add(int(it.nutrition_item_id))
        
        # Nutrition likes
        like_query = db.query(Interaction).filter(
            Interaction.user_id == user.id,
            Interaction.nutrition_item_id != None,
            Interaction.event_type == 'like'
        ).all()
        for it in like_query:
            if it.nutrition_item_id:
                liked_nutrition_ids.add(int(it.nutrition_item_id))
        
        # Fitness dislikes
        fit_dislike_query = db.query(Interaction).filter(
            Interaction.user_id == user.id,
            Interaction.fitness_item_id != None,
            Interaction.event_type == 'dislike'
        ).all()
        for it in fit_dislike_query:
            if it.fitness_item_id:
                disliked_fitness_ids.add(int(it.fitness_item_id))
        
        # Fitness likes
        fit_like_query = db.query(Interaction).filter(
            Interaction.user_id == user.id,
            Interaction.fitness_item_id != None,
            Interaction.event_type == 'like'
        ).all()
        for it in fit_like_query:
            if it.fitness_item_id:
                liked_fitness_ids.add(int(it.fitness_item_id))
    except Exception:
        pass
    
    # **Compute user-based collaborative filtering: find similar users and their liked items**
    similar_user_likes = set()
    try:
        all_users = db.query(User).all()
        user_attrs = (user.age, user.weight, user.height, user.activity_level, user.health_goals)
        user_dists = []
        for other_user in all_users:
            if other_user.id == user.id:
                continue
            other_attrs = (other_user.age, other_user.weight, other_user.height, other_user.activity_level, other_user.health_goals)
            # Compute simple Euclidean distance (normalize numeric attrs)
            try:
                age_sim = 1.0 / (1.0 + abs(user_attrs[0] - other_attrs[0]) / 10.0)
                weight_sim = 1.0 / (1.0 + abs(user_attrs[1] - other_attrs[1]) / 20.0)
                height_sim = 1.0 / (1.0 + abs(user_attrs[2] - other_attrs[2]) / 10.0)
                activity_match = 1.0 if user_attrs[3] == other_attrs[3] else 0.5
                goal_match = 1.0 if user_attrs[4] == other_attrs[4] else 0.5
                similarity = (age_sim + weight_sim + height_sim + activity_match + goal_match) / 5.0
                if similarity > 0.6:  # Threshold for "similar" user
                    user_dists.append((similarity, other_user.id))
            except Exception:
                pass
        
        # Get liked items from top 3 similar users
        user_dists.sort(reverse=True)
        for _, sim_user_id in user_dists[:3]:
            likes = db.query(Interaction).filter(
                Interaction.user_id == sim_user_id,
                Interaction.nutrition_item_id != None,
                Interaction.event_type == 'like'
            ).all()
            for like_it in likes:
                if like_it.nutrition_item_id:
                    similar_user_likes.add(int(like_it.nutrition_item_id))
    except Exception:
        pass
    
    # Filter out disliked items from recommendations
    nutrition_items = [it for it in nutrition_items if it.id not in disliked_nutrition_ids]
    fitness_items = [it for it in fitness_items if it.id not in disliked_fitness_ids]
    
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

    # Provide reasonable default daily targets for fiber and magnesium so
    # the nutrition scorer and meal planner can personalize for these nutrients.
    try:
        # Fiber: general adult target ~25-30 g/day
        if not nutrition_targets.get('fiber_g'):
            nutrition_targets['fiber_g'] = 30.0 if (gender and gender[0] == 'M') else 25.0
        # Magnesium: typical adult targets (mg/day)
        if not nutrition_targets.get('magnesium_mg') and not nutrition_targets.get('magnesium'):
            if gender and gender[0] == 'M':
                nutrition_targets['magnesium_mg'] = 400.0
            else:
                nutrition_targets['magnesium_mg'] = 310.0
    except Exception:
        # fall back to safe defaults
        nutrition_targets.setdefault('fiber_g', 25.0)
        nutrition_targets.setdefault('magnesium_mg', 350.0)
    
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

    # Generate recommendations lazily
    nutrition_recs = None
    fitness_recs = None
    nutr_rec = None  # Initialize to None; may be set below if ML is enabled
    fit_rec = None
    
    # Only generate nutrition recommendations if needed
    if rec_type in ('nutrition', 'all') and not DISABLE_ML:
        try:
            # Attempt to load persisted models (if available) to provide real collaborative scores.
            from recsys_app.services.training import load_models
            loaded = load_models()
            nutr_rec = loaded.get('nutrition') if isinstance(loaded, dict) else None
            # If loading failed or models are not present, fall back to fresh instances
            if nutr_rec is None:
                from ..recommenders.nutrition.recommender import NutritionRecommender
                nutr_rec = NutritionRecommender()
            # Fetch user's positive and negative nutrition interactions to inform item-based similarity
            try:
                pos_q = db.query(Interaction).filter(
                    Interaction.user_id == user.id,
                    Interaction.nutrition_item_id != None
                )
                pos_items = []
                neg_items = []
                for it in pos_q.all():
                    # consider explicit accepts/clicks or positive ratings
                    try:
                        rated = float(getattr(it, 'rating', 0) or 0)
                    except Exception:
                        rated = 0.0
                    et = getattr(it, 'event_type', None)
                    if et in ('accept', 'click') or rated > 0:
                        if it.nutrition_item_id:
                            pos_items.append(int(it.nutrition_item_id))
                    # negative signals: explicit 'dislike' or negative rating
                    if et == 'dislike' or rated < 0:
                        if it.nutrition_item_id:
                            neg_items.append(int(it.nutrition_item_id))
                # attach to recommender instance for use during generate
                if pos_items:
                    try:
                        nutr_rec.user_positive_item_ids = list(set(pos_items))
                    except Exception:
                        pass
                if neg_items:
                    try:
                        nutr_rec.user_negative_item_ids = list(set(neg_items))
                    except Exception:
                        pass
            except Exception:
                pass
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
    
    # Only generate fitness recommendations if needed
    if rec_type in ('fitness', 'all') and not DISABLE_ML:
        try:
            from recsys_app.services.training import load_models
            loaded = load_models()
            fit_rec = loaded.get('fitness') if isinstance(loaded, dict) else None
            # If loading failed or models are not present, fall back to fresh instances
            if fit_rec is None:
                from ..recommenders.fitness.recommender import FitnessRecommender
                fit_rec = FitnessRecommender()
            # Fetch user's positive and negative fitness interactions
            try:
                fit_q = db.query(Interaction).filter(
                    Interaction.user_id == user.id,
                    Interaction.fitness_item_id != None
                )
                fit_pos_items = []
                fit_neg_items = []
                for it in fit_q.all():
                    # consider explicit accepts/clicks or positive ratings
                    try:
                        rated = float(getattr(it, 'rating', 0) or 0)
                    except Exception:
                        rated = 0.0
                    et = getattr(it, 'event_type', None)
                    if et in ('accept', 'click') or rated > 0:
                        if it.fitness_item_id:
                            fit_pos_items.append(int(it.fitness_item_id))
                    # negative signals: explicit 'dislike' or negative rating
                    if et == 'dislike' or rated < 0:
                        if it.fitness_item_id:
                            fit_neg_items.append(int(it.fitness_item_id))
                # attach to recommender instance for use during generate
                if fit_pos_items:
                    try:
                        fit_rec.user_positive_item_ids = list(set(fit_pos_items))
                    except Exception:
                        pass
                if fit_neg_items:
                    try:
                        fit_rec.user_negative_item_ids = list(set(fit_neg_items))
                    except Exception:
                        pass
            except Exception:
                pass
            fitness_recs = fit_rec.generate_recommendations(user_df.iloc[0].to_dict(), top_k=top_k)
        except Exception:
            fitness_recs = fitness_df.head(top_k)
    elif rec_type in ('fitness', 'all'):
        # ML disabled — use content-only fallback
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
        keys = ['id', 'name', 'type', 'level', 'equipment', 'bodypart', 'hybrid_score']
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

    # Final sanitization:
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

    # Build a meal plan using nutrition utilities if nutrition_targets are present and nutrition items exist
    meal_plan_output = None
    if rec_type in ('nutrition', 'all') and len(nutrition_df) > 0:
        try:
            from ..recommenders.nutrition.utils import generate_daily_meal_plan
            # Prepare liked_item_ids for meal plan generation
            liked_item_ids = liked_nutrition_ids if liked_nutrition_ids else set()
            # Define default meal calorie distribution
            # Breakfast 25%, Lunch 40%, Dinner 35%
            meal_cal_dist = {
                'Breakfast': 0.25,
                'Lunch': 0.40,
                'Dinner': 0.35
            }
            
            # Build meal_context_map from nutrition_df for contextual recommendations
            meal_context_map = {}
            try:
                for _, row in nutrition_df.iterrows():
                    item_id = row.get('id')
                    item_context = row.get('meal_context', '')
                    if item_id and item_context:
                        meal_context_map[int(item_id)] = str(item_context).lower()
            except Exception:
                pass
            
            meal_plan = generate_daily_meal_plan(
                user_df.iloc[0].to_dict(),
                nutrition_df,
                nutr_rec,                  # or None
                meals=['Breakfast', 'Lunch', 'Dinner'],
                items_per_meal=4,
                top_k_candidates=120,
                nutrition_targets=nutrition_targets,
                meals_per_day=3,
                diversity_overlap_threshold=0.5,
                liked_item_ids=liked_item_ids,  # Force liked items into meal plan
                meal_cal_dist=meal_cal_dist, 
                include_snacks=True,  # enable Snacks slot to increase item diversity
                similar_user_likes=similar_user_likes,  # Pass collaborative filtering likes
                meal_context_map=meal_context_map  # Pass meal context for contextual boosting
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
                        'carbohydrates': it.get('carbohydrates'),
                        'fiber': it.get('fiber') if it.get('fiber') is not None else it.get('fiber_val'),
                        'magnesium': it.get('magnesium') if it.get('magnesium') is not None else it.get('magnesium_val'),
                        'serving_multiplier': it.get('serving_multiplier', 1.0),  # portion scaling
                        'reason': it.get('reason', 'recommended')  # why recommended
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
    EVENT_WEIGHTS = {
        'accept': 1.0,
        'click': 0.6,
        'like': 1.0,
        'impression': 0.05,
        'skip': -0.4,
        'dislike': -0.8
    }
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

    LAMBDA_INTERACTION = 4.0
    LAMBDA_LIKED = 10.0  # Strong boost for explicitly liked items

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
            # Add strong boost for explicitly liked items
            if iid in liked_item_ids:
                boost += LAMBDA_LIKED
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
    daily_fiber = None
    daily_magnesium = None
    daily_magnesium_pct = None
    magnesium_warning = None
    percent_of_goal = None
    try:
        if meal_plan_output:
            dp = 0.0
            dc = 0.0
            dfiber = 0.0
            dmag = 0.0
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
                    try:
                        dfiber += float(it.get('fiber') or 0.0)
                    except Exception:
                        pass
                    try:
                        dmag += float(it.get('magnesium') or 0.0)
                    except Exception:
                        pass
            daily_protein = float(dp)
            daily_calories = float(dc)
            daily_fiber = float(dfiber)
            daily_magnesium = float(dmag)
            # compute percent of RDA based on gender/age defaults used earlier
            try:
                rda_mag = 400.0 if (gender and gender[0] == 'M') else 310.0
                daily_magnesium_pct = float((daily_magnesium / max(1.0, rda_mag)) * 100.0)
                # warning if > 300% of RDA (very high)
                magnesium_warning = True if daily_magnesium_pct > 300.0 else False
            except Exception:
                daily_magnesium_pct = None
                magnesium_warning = None
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

    result = RecommendationResponse(
        nutrition_items=nutr_list if rec_type in ('nutrition', 'all') else [],
        fitness_items=fit_list if rec_type in ('fitness', 'all') else [],
        tdee=float(tdee) if tdee is not None else None,
        nutrition_goal=float(nutrition_goal) if nutrition_goal is not None else None,
        bmr=float(bmr) if bmr is not None else None,
        nutrition_targets=nutrition_targets,
        meal_plan=meal_plan_output if rec_type in ('nutrition', 'all') else None,
        daily_protein=daily_protein if rec_type in ('nutrition', 'all') else None,
        daily_calories=daily_calories if rec_type in ('nutrition', 'all') else None,
        percent_of_goal=percent_of_goal if rec_type in ('nutrition', 'all') else None,
        daily_fiber=daily_fiber if rec_type in ('nutrition', 'all') else None,
        daily_magnesium=daily_magnesium if rec_type in ('nutrition', 'all') else None,
        daily_magnesium_pct=daily_magnesium_pct if rec_type in ('nutrition', 'all') else None,
        magnesium_warning=magnesium_warning if rec_type in ('nutrition', 'all') else None
    )
    print(f'[RECOMMENDATION] Returning {len(result.nutrition_items)} nutrition, {len(result.fitness_items)} fitness')
    return result