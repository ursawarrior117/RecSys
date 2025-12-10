"""Utility functions for nutrition recommendations."""
import numpy as np
import pandas as pd

def hybrid_nutrition_recommendations(user: dict, nutrition_data: pd.DataFrame, recommender, top_k: int = 20,
                                     alpha: float = 0.5, nutrition_targets: dict = None,
                                     prot_boost_weight: float = 0.30,
                                     size_penalty_weight: float = 0.35,
                                     size_threshold_frac: float = 0.4,
                                     meals_per_day: int = 3,
                                     diversity_overlap_threshold: float = 0.4,
                                     sim_strength: float = 0.5):
    """Generate hybrid nutrition recommendations combining collaborative and content-based filtering."""
    # Try collaborative predictions, but fall back to content-only if the recommender
    # or its preprocessors are not ready (e.g., not fitted or no model loaded).
    try:
        collab_scores = recommender.predict_scores(user, nutrition_data)
    except Exception:
        # Use uniform/neutral collaborative scores so content-based scoring dominates
        collab_scores = np.zeros(len(nutrition_data))
    
    # Content-based scoring based on nutritional metrics
    content_scores = (
        0.4 * nutrition_data['protein'] / (nutrition_data['calories'] + 1) +
        0.2 * nutrition_data['fiber'] / (nutrition_data['calories'] + 1) -
        0.2 * nutrition_data['fat'] / (nutrition_data['calories'] + 1) -
        0.2 * nutrition_data['sugars'] / (nutrition_data['calories'] + 1)
    )
    
    # Add magnesium boost for sleep issues
    if 'sleep_good' in user and user['sleep_good'] == 0 and 'magnesium' in nutrition_data.columns:
        content_scores += 0.2 * nutrition_data['magnesium'] / (nutrition_data['magnesium'].max() + 1e-8)
    
    # Normalize scores
    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)

    # If the recommender provides positive/negative item ids and item preprocessors,
    # compute an item-based similarity signal and blend it into the content score.
    try:
        if recommender is not None and hasattr(recommender, 'item_preprocessor') and (
            getattr(recommender, 'user_positive_item_ids', None) or getattr(recommender, 'user_negative_item_ids', None)
        ):
            try:
                # Prepare item vectors using recommender's item_feature_cols
                cols = getattr(recommender, 'item_feature_cols', None)
                if cols is not None:
                    nd = nutrition_data.copy()
                    for c in cols:
                        if c not in nd.columns:
                            nd[c] = 0
                    # transform if possible
                    item_vecs = recommender.item_preprocessor.transform(nd[cols])
                    norms = np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-8
                    item_vecs_norm = item_vecs / norms

                    id_list = list(nutrition_data['id'].tolist()) if 'id' in nutrition_data.columns else None
                    pos_idx = []
                    neg_idx = []
                    if getattr(recommender, 'user_positive_item_ids', None) and id_list is not None:
                        pos_idx = [id_list.index(pid) for pid in recommender.user_positive_item_ids if pid in id_list]
                    if getattr(recommender, 'user_negative_item_ids', None) and id_list is not None:
                        neg_idx = [id_list.index(pid) for pid in recommender.user_negative_item_ids if pid in id_list]

                    sim_scores = np.zeros(len(item_vecs_norm))
                    sim_neg = np.zeros(len(item_vecs_norm))
                    if len(pos_idx) > 0:
                        for i in range(len(item_vecs_norm)):
                            sims = item_vecs_norm[pos_idx] @ item_vecs_norm[i]
                            sim_scores[i] = np.max(sims) if len(sims) > 0 else 0.0
                    if len(neg_idx) > 0:
                        for i in range(len(item_vecs_norm)):
                            sims = item_vecs_norm[neg_idx] @ item_vecs_norm[i]
                            sim_neg[i] = np.max(sims) if len(sims) > 0 else 0.0

                    # normalize sim signals
                    if sim_scores.size > 0:
                        smin, smax = sim_scores.min(), sim_scores.max()
                        if smax > smin:
                            sim_scores = (sim_scores - smin) / (smax - smin)
                        else:
                            sim_scores = np.zeros_like(sim_scores)
                    if sim_neg.size > 0:
                        nmin, nmax = sim_neg.min(), sim_neg.max()
                        if nmax > nmin:
                            sim_neg = (sim_neg - nmin) / (nmax - nmin)
                        else:
                            sim_neg = np.zeros_like(sim_neg)

                    # combine positive and negative similarity into a single effect
                    negative_penalty = 0.8
                    sim_effect = sim_scores - negative_penalty * sim_neg
                    # blend sim_effect into content_scores
                    # ensure sim_effect scaled to 0..1 like content_scores
                    if sim_effect.size > 0:
                        se_min, se_max = sim_effect.min(), sim_effect.max()
                        if se_max > se_min:
                            sim_effect = (sim_effect - se_min) / (se_max - se_min)
                        else:
                            sim_effect = np.zeros_like(sim_effect)
                        # mix into content scores
                        content_scores = (content_scores + sim_strength * sim_effect)
                        # renormalize content_scores
                        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
            except Exception:
                pass
    except Exception:
        pass

    # If nutrition targets (daily goals) are provided, nudge content scores
    if nutrition_targets is not None:
        try:
            # continuous adjustments based on per-meal protein and calorie ratios
            tdee = float(nutrition_targets.get('calories', 0))
            protein_goal = float(nutrition_targets.get('protein_g', 0))
            meals = float(meals_per_day or 3)
            per_meal_protein = max(1.0, protein_goal / meals)

            # protein_ratio: fraction of per-meal protein provided by the item
            # clip to [0, 2] so extremely high values don't dominate
            prot_ratio = (nutrition_data['protein'] / (per_meal_protein + 1e-8)).fillna(0.0).clip(0.0, 2.0)

            # protein score scales with prot_ratio (0..2) and is centered so items meeting target get boost ~1
            prot_score = (prot_ratio - 1.0).clip(-1.0, 2.0)

            # calorie ratio relative to per-meal calories; if tdee missing, do not penalize
            if tdee and tdee > 0:
                per_meal_cal = (tdee / meals)
                cal_ratio = (nutrition_data['calories'] / (per_meal_cal + 1e-8)).fillna(0.0)
                # size_penalty scales when cal_ratio exceeds size_threshold_frac
                size_penalty = (cal_ratio / (size_threshold_frac + 1e-8) - 1.0).clip(0.0, 2.0)
            else:
                size_penalty = 0.0

            # apply continuous boosts and penalties
            content_scores = content_scores + prot_boost_weight * prot_score - size_penalty_weight * size_penalty

            # re-normalize content_scores
            content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
        except Exception:
            pass

    # Combine scores
    hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores
    nutrition_data = nutrition_data.copy()
    nutrition_data['hybrid_score'] = hybrid_scores
    nutrition_data = nutrition_data.sort_values('hybrid_score', ascending=False)
    
    # Post-process to enforce realistic variety: greedy select top_k items
    # avoiding near-duplicate food names and oversized items relative to per-meal calories.
    def _tokenize_name(name: str):
        if not name:
            return set()
        name = str(name).lower()
        # simple split, remove short stopwords
        toks = [t for t in name.replace(',', ' ').replace('(', ' ').replace(')', ' ').split() if len(t) > 2]
        return set(toks)

    per_meal_cal = None
    try:
        if nutrition_targets is not None and nutrition_targets.get('calories'):
            cal = float(nutrition_targets.get('calories'))
            if cal > 0 and meals_per_day > 0:
                per_meal_cal = cal / float(meals_per_day)
    except Exception:
        per_meal_cal = None

    selected_rows = []
    selected_tokens = []
    seen_foods = set()

    # compute token frequencies to detect overly-common tokens (e.g., "milk")
    token_freq = {}
    for _, r in nutrition_data.iterrows():
        toks = _tokenize_name(r.get('food', ''))
        for t in toks:
            token_freq[t] = token_freq.get(t, 0) + 1
    common_tokens = set()
    try:
        freq_threshold = max(10, int(0.10 * len(nutrition_data)))
        for t, cnt in token_freq.items():
            if cnt >= freq_threshold:
                common_tokens.add(t)
    except Exception:
        common_tokens = set()
    # allow at most this many items sharing a common token
    max_common_occurrence = 2

    for _, row in nutrition_data.iterrows():
        if len(selected_rows) >= top_k:
            break
        name = row.get('food', '')
        if not name or name in seen_foods:
            continue
        toks = _tokenize_name(name)

        # diversity: avoid items with too much token overlap with already selected items
        too_similar = False
        for prev in selected_tokens:
            if not toks or not prev:
                continue
            overlap = len(toks & prev) / float(len(toks | prev))
            if overlap >= diversity_overlap_threshold:
                too_similar = True
                break
        if too_similar:
            continue

        # common-token cap: avoid picking many items that share the same common token
        skip_common = False
        for ct in (toks & common_tokens):
            # count how many selected items contain this token
            cur_count = sum(1 for st in selected_tokens if ct in st)
            if cur_count >= max_common_occurrence:
                skip_common = True
                break
        if skip_common:
            continue

        # size constraint: avoid items that are larger than a reasonable per-meal portion
        try:
            cals = float(row.get('calories') or 0.0)
            if per_meal_cal is not None and cals > (per_meal_cal * 0.9):
                # If the item is larger than ~90% of per-meal calories, skip it for variety
                continue
        except Exception:
            pass

        # accept item
        selected_rows.append(row)
        selected_tokens.append(toks)
        seen_foods.add(name)

    # If we didn't find enough diverse items, fall back to top-k by score
    if len(selected_rows) < top_k:
        remaining = nutrition_data[~nutrition_data['food'].isin(seen_foods)].head(top_k - len(selected_rows))
        for _, r in remaining.iterrows():
            selected_rows.append(r)

    if not selected_rows:
        return nutrition_data.head(top_k)

    result_df = pd.DataFrame(selected_rows)
    return result_df.reset_index(drop=True)

def generate_daily_meal_plan(user: dict, nutrition_data: pd.DataFrame, recommender, meals=['Breakfast', 'Lunch', 'Dinner'],
                             items_per_meal: int = 1, top_k_candidates: int = 60, nutrition_targets: dict = None,
                             meals_per_day: int = 3, diversity_overlap_threshold: float = 0.5, liked_item_ids: set = None):
    """Create a simple daily meal plan composed of items that together aim to meet daily protein targets.

    Strategy (greedy with category diversity):
    - Use hybrid scores if present, otherwise compute a shortlist via hybrid_nutrition_recommendations.
    - For each meal, pick up to `items_per_meal` items from different WWEIA categories to ensure variety
      (e.g., one dairy, one meat, one vegetable per meal if available).
    - Prefer higher-protein items but avoid repeating items across meals and within meals.
    - Apply strict token-based diversity checks to prevent similar food names.
    """
    if nutrition_data is None or len(nutrition_data) == 0:
        return {m: [] for m in meals}
    
    nd = nutrition_data.copy()

    # ensure we have a scored shortlist
    if 'hybrid_score' not in nd.columns:
        try:
            scored = hybrid_nutrition_recommendations(user, nd, recommender, top_k=top_k_candidates,
                                                      alpha=0.5, nutrition_targets=nutrition_targets,
                                                      meals_per_day=meals_per_day,
                                                      diversity_overlap_threshold=diversity_overlap_threshold)
            nd = scored.copy()
        except Exception:
            # Fallback: sort by protein and use top candidates
            if 'protein' in nd.columns:
                nd = nd.sort_values('protein', ascending=False).head(top_k_candidates)
            else:
                nd = nd.head(top_k_candidates)
    else:
        nd = nd.sort_values('hybrid_score', ascending=False).head(top_k_candidates)
    
    if len(nd) == 0:
        return {m: [] for m in meals}

    def _tokenize_name(name: str):
        if not name:
            return set()
        name = str(name).lower()
        toks = [t for t in name.replace(',', ' ').replace('(', ' ').replace(')', ' ').split() if len(t) > 2]
        return set(toks)

    protein_goal = 0.0
    tdee = None
    try:
        if nutrition_targets is not None:
            protein_goal = float(nutrition_targets.get('protein_g', 0.0) or 0.0)
            tdee = float(nutrition_targets.get('calories') or 0.0) if nutrition_targets.get('calories') else None
    except Exception:
        protein_goal = 0.0
        tdee = None

    # per-meal protein target (initial allocation); will be adapted during greedy fill
    per_meal_protein = max(1.0, protein_goal / float(meals_per_day or 3)) if protein_goal > 0 else None

    # candidate list as dicts for easy consumption
    candidates = []
    for idx, (_, r) in enumerate(nd.iterrows()):
        d = r.to_dict()
        d['tokens'] = _tokenize_name(d.get('food') or d.get('name') or '')
        d['protein_val'] = float(d.get('protein') or 0.0)
        d['calories_val'] = float(d.get('calories') or 0.0)
        d['category'] = d.get('WWEIA Category number') or d.get('WWEIA Category description') or 'unknown'
        # ensure hybrid_score is present and numeric
        try:
            d['hybrid_score'] = float(d.get('hybrid_score', 0.0))
        except Exception:
            d['hybrid_score'] = 0.0
        # Create a unique ID using Food code or row index
        d['_meal_planner_id'] = d.get('Food code', str(idx))
        candidates.append(d)
    
    if len(candidates) == 0:
        print(f'[Meal Planner] Warning: no candidates available from {len(nd)} nutrition items')
        return {m: [] for m in meals}

    used_ids = set()
    plan = {}

    # daily totals tracking
    daily_protein = 0.0
    daily_cal = 0.0

    # rank candidates by calories first (to fill daily goal), then protein-to-calorie ratio
    # This ensures we select calorie-dense items while maintaining protein balance
    # If liked_item_ids provided, prioritize them in ranking
    def rank_candidates(avail):
        if liked_item_ids:
            return sorted(avail,
                          key=lambda x: (
                              x.get('id') not in liked_item_ids,  # False (liked) sorts before True (not liked)
                              -x.get('calories_val', 0.0),  # negate for descending
                              -x.get('protein_val', 0.0) / (x.get('calories_val', 1.0) + 1e-8),
                              -x.get('hybrid_score', 0.0)
                          ))
        else:
            return sorted(avail,
                          key=lambda x: (x.get('calories_val', 0.0), x.get('protein_val', 0.0) / (x.get('calories_val', 1.0) + 1e-8), x.get('hybrid_score', 0.0)),
                          reverse=True)

    # maximum items per meal hard cap to avoid runaway plans
    # Use items_per_meal to set a reasonable cap (usually 2-4 items per meal)
    max_items_per_meal = items_per_meal + 2

    remaining_protein = protein_goal
    
    # Define target_cal early for use in the loop
    target_cal = tdee if tdee else None

    for i, meal in enumerate(meals):
        selected = []
        selected_tokens = []

        # compute per-meal calorie target (distribute calories evenly across meals)
        # Allow some flexibility (0.8x to 1.2x average per meal)
        per_meal_cal = (target_cal / float(meals_per_day or 3)) if target_cal and target_cal > 0 else None
        min_meal_cal = (per_meal_cal * 0.8) if per_meal_cal else 0
        ideal_meal_cal = (per_meal_cal * 1.0) if per_meal_cal else 0
        max_meal_cal = (per_meal_cal * 1.5) if per_meal_cal else float('inf')
        
        # For protein, just use a fixed per-meal average (not adaptive, to avoid overshooting)
        fixed_meal_protein = (protein_goal / float(meals_per_day or 3)) if protein_goal > 0 else None

        # available candidates excluding used ids
        available = [c for c in candidates if c.get('_meal_planner_id') not in used_ids]
        ranked = rank_candidates(available)

        # Greedily add items until meal calorie target met or cap reached
        for c in ranked:
            if len(selected) >= max_items_per_meal:
                break

            current_meal_cal = sum(x['calories_val'] for x in selected)
            current_meal_prot = sum(x['protein_val'] for x in selected)
            
            # Don't be too aggressive about stopping early
            # Continue if we're below 80% of calorie target
            if current_meal_cal < ideal_meal_cal * 0.8:
                pass  # Keep adding items
            elif len(selected) >= items_per_meal:
                # If we have minimum items and have some reasonable nutrition, we can stop
                # But be lenient: only stop if we have at least items_per_meal items
                if current_meal_cal >= ideal_meal_cal * 0.7:
                    break
            
            # Stop if adding this item would exceed protein cap significantly (allow 30% overage)
            if (fixed_meal_protein is not None and 
                current_meal_prot + c['protein_val'] > fixed_meal_protein * 1.3):
                continue

            # diversity check
            too_similar = False
            for prev in selected_tokens:
                if not c['tokens'] or not prev:
                    continue
                overlap = len(c['tokens'] & prev) / float(len(c['tokens'] | prev))
                if overlap >= diversity_overlap_threshold:
                    too_similar = True
                    break
            if too_similar:
                continue

            # size check: skip very large items unless meal is empty
            if per_meal_cal and c['calories_val'] > max_meal_cal and len(selected) > 0:
                continue

            # choose item
            selected.append(c)
            selected_tokens.append(c['tokens'])
            used_ids.add(c.get('_meal_planner_id'))

        # Build output items for this meal
        out_items = []
        for it in selected[:max_items_per_meal]:
            out_items.append({
                'id': it.get('id'),
                'food': it.get('food') or it.get('name'),
                'calories': it.get('calories_val'),
                'protein': it.get('protein_val'),
                'fat': it.get('fat') if 'fat' in it else None,
                'carbohydrates': it.get('carbohydrates') if 'carbohydrates' in it else None
            })

        # update daily totals
        daily_protein += sum(x['protein_val'] for x in selected[:max_items_per_meal])
        daily_cal += sum(x['calories_val'] for x in selected[:max_items_per_meal])

        plan[meal] = out_items

    # After meals, try to fill calorie gap (prioritize calories, then protein balance)
    calorie_gap = (target_cal - daily_cal) if target_cal and target_cal > 0 else 0
    
    # Only top-up if we're below 90% of calorie goal
    if target_cal and calorie_gap > target_cal * 0.1:
        leftovers = [c for c in candidates if c.get('_meal_planner_id') not in used_ids]
        
        # Sort by calories (to fill calorie gap first), then avoid protein items if possible
        items_added = 0
        last_meal = meals[-1]
        
        for c in sorted(leftovers, key=lambda x: (-x.get('calories_val', 0.0), x.get('protein_val', 0.0))):
            # Hard stop if calorie goal is met
            if daily_cal >= target_cal * 0.95:
                break
            
            # Don't add items if calorie goal is met
            if daily_cal >= target_cal:
                break
            
            # Limit items added in top-up phase (allow up to 10 to fill calories)
            if items_added >= 10:
                break
            
            # append to last meal
            plan[last_meal].append({
                'id': c.get('id'),
                'food': c.get('food') or c.get('name'),
                'calories': c.get('calories_val'),
                'protein': c.get('protein_val'),
                'fat': c.get('fat') if 'fat' in c else None,
                'carbohydrates': c.get('carbohydrates') if 'carbohydrates' in c else None
            })
            daily_protein += c.get('protein_val', 0.0)
            daily_cal += c.get('calories_val', 0.0)
            items_added += 1

    return plan