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
                                     sim_strength: float = 0.5,
                                     fiber_boost_weight: float = 0.10,
                                     magnesium_boost_weight: float = 0.03,
                                     liked_item_ids: set = None,
                                     similar_user_likes: set = None,
                                     meal_context: str = None):
    
    """Generate hybrid nutrition recommendations combining collaborative and content-based filtering."""
    # Try collaborative predictions, but fall back to content-only if the recommender or its preprocessors are not ready
    try:
        collab_scores = recommender.predict_scores(user, nutrition_data)
    except Exception:
        # Use uniform/neutral collaborative scores so content-based scoring dominates
        collab_scores = np.zeros(len(nutrition_data))
    
    # Content-based scoring based on nutritional metrics
    content_scores = (
        0.15 * nutrition_data['protein'] / (nutrition_data['calories'] + 1) +
        0.25 * nutrition_data['fiber'] / (nutrition_data['calories'] + 1) -
        0.2 * nutrition_data['fat'] / (nutrition_data['calories'] + 1) -
        0.2 * nutrition_data['sugars'] / (nutrition_data['calories'] + 1)
    )
    
    # Add magnesium boost for sleep issues
    if 'sleep_good' in user and user['sleep_good'] == 0 and 'magnesium' in nutrition_data.columns:
        content_scores += 0.2 * nutrition_data['magnesium'] / (nutrition_data['magnesium'].max() + 1e-8)
    
    # Normalize scores
    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)

    # If the recommender provides positive/negative item ids and item preprocessors, compute an item-based similarity signal and blend it into the content score.
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

            # clip protein ratio to [0, 2] so extremely high values don't dominate
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

            # Fiber & magnesium personalization: if user provided daily targets for fiber/magnesium,
            # nudge items that help meet per-meal targets.
            try:
                # detect fiber column
                fiber_col = None
                mag_col = None
                for c in nutrition_data.columns:
                    lc = str(c).lower()
                    if 'fiber' in lc and fiber_col is None:
                        fiber_col = c
                    if 'magnesium' in lc and mag_col is None:
                        mag_col = c

                # per-meal fiber target (grams)
                per_meal_fiber = None
                if nutrition_targets.get('fiber_g'):
                    per_meal_fiber = max(0.1, float(nutrition_targets.get('fiber_g')) / float(meals_per_day or 3))

                # per-meal magnesium target (mg)
                per_meal_mag = None
                # accept either 'magnesium_mg' or 'magnesium' in nutrition_targets
                if nutrition_targets.get('magnesium_mg'):
                    per_meal_mag = max(1.0, float(nutrition_targets.get('magnesium_mg')) / float(meals_per_day or 3))
                elif nutrition_targets.get('magnesium'):
                    per_meal_mag = max(1.0, float(nutrition_targets.get('magnesium')) / float(meals_per_day or 3))

                # compute fiber score
                if per_meal_fiber is not None and fiber_col is not None:
                    try:
                        fiber_ratio = (nutrition_data[fiber_col] / (per_meal_fiber + 1e-8)).fillna(0.0).clip(0.0, 3.0)
                        fiber_score = (fiber_ratio - 1.0).clip(-1.0, 2.0)
                        content_scores = content_scores + fiber_boost_weight * fiber_score
                    except Exception:
                        pass

                # compute magnesium score
                if per_meal_mag is not None and mag_col is not None:
                    try:
                        mag_ratio = (nutrition_data[mag_col] / (per_meal_mag + 1e-8)).fillna(0.0).clip(0.0, 3.0)
                        mag_score = (mag_ratio - 1.0).clip(-1.0, 2.0)
                        content_scores = content_scores + magnesium_boost_weight * mag_score
                    except Exception:
                        pass
            except Exception:
                pass

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
    # avoiding duplicate food names and oversized items
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
            cur_count = sum(1 for st in selected_tokens if ct in st)
            if cur_count >= max_common_occurrence:
                skip_common = True
                break
        if skip_common:
            continue

        # item size constraint
        try:
            cals = float(row.get('calories') or 0.0)
            if per_meal_cal is not None and cals > (per_meal_cal * 0.9):
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

def _knapsack_select_meal(candidates: list, capacity_cal: float, max_items: int = 4, protein_goal: float = 20.0, beta: float = 0.5, calories_weight: float = 0.15):
    if not candidates or capacity_cal <= 0:
        return []
    
    # Discretize capacity to reduce DP table size
    cal_step = 10  # 10 kcal granularity
    cap_bins = max(1, int(capacity_cal / cal_step))
    
    # Item value: cap per-item protein contribution so many small high-protein items. Value mixes capped protein + calories + hybrid score.
    max_score = max([c.get('hybrid_score', 0.0) for c in candidates] + [1.0])
    items_with_value = []
    # cap per-item protein contribution to a fraction of per-meal target
    cap_per_item = max(1.0, (protein_goal * 0.9) / float(max_items))
    for c in candidates:
        score = c.get('hybrid_score', 0.0) / (max_score + 1e-8)
        prot = c.get('protein_val', 0.0)
        prot_contrib = min(prot, cap_per_item)
        # include fiber and magnesium contributions if present (small weights)
        fiber_contrib = c.get('fiber_val', 0.0)
        # cap per-item magnesium contribution (avoid unrealistic large contributions)
        # reduce cap so a single item cannot dominate magnesium totals
        magnesium_contrib = min(float(c.get('magnesium_val', 0.0) or 0.0), 50.0)
        # diminishing returns: use sqrt for high magnesium values so many-high-mag items don't dominate
        try:
            magnesium_effect = (magnesium_contrib ** 0.5)
        except Exception:
            magnesium_effect = magnesium_contrib
        # value blends limited protein contribution, calories, fiber/magnesium, and model score
        # increase fiber weight to encourage a broader set of fiber-rich items
        value = (
            0.25 * prot_contrib
            + calories_weight * c.get('calories_val', 0.0)
            + 0.08 * fiber_contrib
            + 0.005 * magnesium_effect
            + beta * score
        )
        items_with_value.append((value, c))
    
    # Sort by value descending for greedy initialization
    items_with_value.sort(key=lambda x: x[0], reverse=True)
    
    # DP: dp[capacity_index][item_count] = best value achievable
    # To save memory, only track best value and reconstruct items
    # Actually simpler: use greedy knapsack since K is small (<=60 items, max 4 per meal)
    selected = []
    used_cal = 0.0
    for value, item in items_with_value:
        if len(selected) >= max_items:
            break
        item_cal = item.get('calories_val', 0.0)
        # prefer items that help reach capacity; allow small overage but avoid underfilling
        if used_cal + item_cal <= capacity_cal * 1.1:
            selected.append(item)
            used_cal += item_cal
    
    return selected


def _select_meal_by_combinations(candidates: list, capacity_cal: float, max_items: int = 4, protein_goal: float = 20.0, beta: float = 0.5, calories_weight: float = 0.15, candidate_limit: int = 30):
    """Try combinations of candidates (limited by candidate_limit) to pick the highest-value set up to max_items.

    This is an exact (brute-force) search over combinations and should be used with small candidate pools (e.g. <=30).
    """
    import itertools
    if not candidates or capacity_cal <= 0:
        return []

    # limit candidates for combinatorial search
    cand = candidates[:max(1, min(len(candidates), candidate_limit))]

    # compute per-item values similarly to knapsack
    max_score = max([c.get('hybrid_score', 0.0) for c in cand] + [1.0])
    items_with_value = []
    cap_per_item = max(1.0, (protein_goal * 0.9) / float(max_items))
    for c in cand:
        score = c.get('hybrid_score', 0.0) / (max_score + 1e-8)
        prot = c.get('protein_val', 0.0)
        prot_contrib = min(prot, cap_per_item)
        fiber_contrib = c.get('fiber_val', 0.0)
        magnesium_contrib = min(float(c.get('magnesium_val', 0.0) or 0.0), 50.0)
        try:
            magnesium_effect = (magnesium_contrib ** 0.5)
        except Exception:
            magnesium_effect = magnesium_contrib
        value = (
            0.25 * prot_contrib
            + calories_weight * c.get('calories_val', 0.0)
            + 0.08 * fiber_contrib
            + 0.005 * magnesium_effect
            + beta * score
        )
        items_with_value.append((value, c))

    # prepare mapping from item to value
    val_map = {c.get('_meal_planner_id'): v for v, c in items_with_value}
    id_map = {c.get('_meal_planner_id'): c for _, c in items_with_value}
    best_combo = []
    best_value = -1e9

    # search combos sizes 1..max_items
    for k in range(1, max_items + 1):
        # avoid too many combinations
        combos = itertools.combinations(items_with_value, k)
        for combo in combos:
            total_cal = sum(item[1].get('calories_val', 0.0) for item in combo)
            if total_cal > capacity_cal * 1.1:
                continue
            total_value = sum(item[0] for item in combo)
            # slight preference for fuller meals (closer to capacity)
            fullness_score = -abs((total_cal / max(1.0, capacity_cal)) - 0.8)
            total_value = total_value + 0.1 * fullness_score
            if total_value > best_value:
                best_value = total_value
                best_combo = [item[1] for item in combo]

    return best_combo


def _scale_portions_for_protein(meal_items: list, target_protein: float, capacity_cal: float = None):
    if not meal_items:
        return []
    
    total_protein = sum(it.get('protein_val', 0.0) for it in meal_items)
    if total_protein <= 0:
        return [(it, 1.0) for it in meal_items]
    
    # Compute a uniform multiplier across all items to meet target but do not exceed calorie budget
    total_cal = sum(it.get('calories_val', 0.0) for it in meal_items)
    # basic multiplier to meet protein
    prot_mult = target_protein / total_protein if total_protein > 0 else 1.0
    # calorie limiter multiplier (do not exceed capacity)
    if capacity_cal and total_cal > 0:
        cal_mult = capacity_cal / total_cal
    else:
        cal_mult = float('inf')

    # Cap at 1.2x to prevent excessive scaling and unrealistic portions
    multiplier = min(1.2, max(1.0, prot_mult, 1.0))
    # but ensure multiplier doesn't exceed calorie allowance
    if cal_mult < multiplier:
        multiplier = max(1.0, cal_mult)

    # final safety clamp
    multiplier = min(1.2, max(1.0, multiplier))

    return [(it, multiplier) for it in meal_items]


def generate_daily_meal_plan(user: dict, nutrition_data: pd.DataFrame, recommender, meals=['Breakfast', 'Lunch', 'Dinner'],
                             items_per_meal: int = 2, top_k_candidates: int = 80, nutrition_targets: dict = None,
                             meals_per_day: int = 3, diversity_overlap_threshold: float = 0.5, liked_item_ids: set = None,
                             meal_cal_dist: dict = None, include_snacks: bool = False, similar_user_likes: set = None,
                             meal_context_map: dict = None, use_greedy_knapsack: bool = True, combination_candidate_limit: int = 30):
    """
    Create a daily meal plan with knapsack-optimized per-meal selection and portion scaling.

    Strategy:
    - Use hybrid scores if present, otherwise compute a shortlist via hybrid_nutrition_recommendations.
    - For each meal, apply knapsack selection to maximize protein (+ hybrid score) subject to per-meal calorie cap.
    - Scale portions (1x to 2x) to meet per-meal protein targets.
    - Enforce diversity via token overlap checks.
    - Liked items are prioritized in candidate ranking.
    
    Args:
        meal_cal_dist: dict of meal -> fraction (e.g., {'Breakfast': 0.25, 'Lunch': 0.4, 'Dinner': 0.35})
        include_snacks: bool, if True add a Snacks meal slot
    """
    if nutrition_data is None or len(nutrition_data) == 0:
        return {m: [] for m in meals}
    
    nd = nutrition_data.copy()

    # **Filter out synthetic/supplement/concentrate items** 
    # These have extreme protein density and skew recommendations toward unrealistic meals
    # Include: protein powders, supplements, meal replacements, dried/concentrated items with protein > 40g/100cal
    synthetic_keywords = ['powder', 'supplement', 'mix', 'shake', 'energy', 'diet product', 'textured', 
                          'dried', 'bran', 'yeast', 'seaweed']
    
    # Remove by keyword
    if 'food' in nd.columns:
        food_names_lower = nd['food'].astype(str).str.lower()
        is_synthetic = food_names_lower.str.contains('|'.join(synthetic_keywords), case=False, na=False)
        nd = nd[~is_synthetic].copy()
    
    # Also filter by protein-to-calorie ratio: items with >0.3g protein per calorie are unrealistic
    # (normal foods have 0.05-0.15g protein/cal)
    if 'protein' in nd.columns and 'calories' in nd.columns:
        protein_ratio = nd['protein'] / (nd['calories'] + 1)
        nd = nd[protein_ratio <= 0.2].copy()  # Cap at 0.2 g protein per calorie

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
        # include fiber and magnesium values for downstream display
        # be robust to different column namings (e.g., 'fiber', 'fiber_g', 'Fiber, total dietary (g)')
        fiber_val = 0.0
        magnesium_val = 0.0
        try:
            # scan for keys that look like fiber or magnesium (case-insensitive)
            for k, v in d.items():
                if v is None:
                    continue
                lk = str(k).lower()
                if 'fiber' in lk and fiber_val == 0.0:
                    try:
                        fiber_val = float(v)
                    except Exception:
                        pass
                if 'magnesium' in lk and magnesium_val == 0.0:
                    try:
                        magnesium_val = float(v)
                    except Exception:
                        pass
        except Exception:
            pass
        d['fiber_val'] = fiber_val or 0.0
        d['magnesium_val'] = magnesium_val or 0.0
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

    # rank candidates by likes, score, then protein ratio
    # If liked_item_ids provided, prioritize them in ranking
    def rank_candidates(avail):
        if liked_item_ids or similar_user_likes:
            return sorted(avail,
                          key=lambda x: (
                              x.get('id') not in liked_item_ids if liked_item_ids else False,  # False (liked) sorts before True (not liked)
                              x.get('id') not in similar_user_likes if similar_user_likes else False,  # False (collab liked) sorts before True
                              -x.get('hybrid_score', 0.0)
                          ))
        else:
            return sorted(avail,
                          key=lambda x: (-x.get('hybrid_score', 0.0)))

    # Set default meal calorie distribution if not provided
    if meal_cal_dist is None:
        meal_cal_dist = {m: 1.0 / float(len(meals)) for m in meals}
    
    # Compute per-meal calorie and protein targets
    target_cal = tdee if tdee else 2000.0
    per_meal_cal_targets = {}
    per_meal_protein_targets = {}
    for meal in meals:
        frac = meal_cal_dist.get(meal, 1.0 / float(len(meals)))
        per_meal_cal_targets[meal] = target_cal * frac
        # Cap per-meal protein to realistic levels (20-30g per meal typical)
        # Reduce excessive protein targets: use much lower default (10g instead of 12g) and stronger cap (25g instead of 32g)
        target_prot = protein_goal * frac if protein_goal > 0 else 10.0
        per_meal_protein_targets[meal] = min(target_prot, 25.0)  # Hard cap at 25g per meal
    
    used_ids = set()
    plan = {}
    daily_protein = 0.0
    daily_cal = 0.0
    
    # **Stage A: Per-meal knapsack selection + portion scaling**
    for meal in meals:
        # Get available candidates (not yet used)
        available = [c for c in candidates if c.get('_meal_planner_id') not in used_ids]
        if not available:
            plan[meal] = []
            continue
        
        # Rank available candidates (prioritize likes, high score, high protein ratio)
        ranked = rank_candidates(available)
        
        # Apply diversity filter to reduce token overlap
        diverse_candidates = []
        diverse_tokens = []
        # compute per-meal magnesium cap (use nutrition_targets if provided)
        per_meal_mag_limit = None
        try:
            if nutrition_targets is not None:
                mag_day = nutrition_targets.get('magnesium_mg') or nutrition_targets.get('magnesium')
                if mag_day:
                    per_meal_mag_limit = float(mag_day) / float(meals_per_day or 3) * 1.2  # allow 20% headroom per meal
        except Exception:
            per_meal_mag_limit = None

        for c in ranked:
            too_similar = False
            for prev_tok in diverse_tokens:
                if not c['tokens'] or not prev_tok:
                    continue
                overlap = len(c['tokens'] & prev_tok) / float(len(c['tokens'] | prev_tok) + 1e-8)
                if overlap >= diversity_overlap_threshold:
                    too_similar = True
                    break
            if too_similar:
                continue
            # if per-meal magnesium limit is set, avoid candidates that alone exceed it
            try:
                if per_meal_mag_limit is not None and float(c.get('magnesium_val', 0.0) or 0.0) > per_meal_mag_limit:
                    continue
            except Exception:
                pass
            diverse_candidates.append(c)
            diverse_tokens.append(c['tokens'])
        
        # Use knapsack to select best items for this meal
        capacity = per_meal_cal_targets[meal]
        if use_greedy_knapsack:
            selected_items = _knapsack_select_meal(
                diverse_candidates,
                capacity_cal=capacity,
                max_items=items_per_meal + 1,
                protein_goal=per_meal_protein_targets[meal],
                beta=0.5,  # balance protein and hybrid score
                calories_weight=0.15
            )
        else:
            # combinations-based selection (exact search on a limited candidate pool)
            selected_items = _select_meal_by_combinations(
                diverse_candidates,
                capacity_cal=capacity,
                max_items=items_per_meal + 1,
                protein_goal=per_meal_protein_targets[meal],
                beta=0.5,
                calories_weight=0.15,
                candidate_limit=combination_candidate_limit
            )
        
        if not selected_items:
            plan[meal] = []
            continue
        
        # Compute portion multipliers to meet per-meal protein target (respecting calorie capacity)
        scaled_items = _scale_portions_for_protein(selected_items, per_meal_protein_targets[meal], capacity_cal=capacity)
        
        # Build output for this meal
        out_items = []
        for item, multiplier in scaled_items:
            serving_text = f" x{multiplier:.1f}" if multiplier > 1.05 else ""
            
            # Compute reason tags
            reasons = []
            if liked_item_ids and item.get('id') in liked_item_ids:
                reasons.append('liked')
            if similar_user_likes and item.get('id') in similar_user_likes:
                reasons.append('similar_to_users_like_you')
            
            # Contextual reason (meal type match)
            if meal_context_map:
                item_context = meal_context_map.get(item.get('id'), '')
                if item_context and meal.lower() in str(item_context).lower():
                    reasons.append(f'fits_{meal.lower()}')
            
            # Health goal reason (high protein for muscle gain)
            health_goals = user.get('health_goals', '')
            if 'muscle' in str(health_goals).lower() and item.get('protein_val', 0.0) > 20:
                reasons.append('supports_muscle_gain')
            elif 'weight_loss' in str(health_goals).lower() and item.get('protein_val', 0.0) > 15:
                reasons.append('supports_weight_loss')
            
            # Default if no specific reasons
            if not reasons:
                reasons.append('recommended')
            
            out_items.append({
                'id': item.get('id'),
                'food': (item.get('food') or item.get('name') or '') + serving_text,
                'calories': item.get('calories_val', 0.0) * multiplier,
                'protein': item.get('protein_val', 0.0) * multiplier,
                'fat': (item.get('fat', 0.0) or 0.0) * multiplier if 'fat' in item else None,
                'carbohydrates': (item.get('carbohydrates', 0.0) or 0.0) * multiplier if 'carbohydrates' in item else None,
                'fiber': (item.get('fiber_val', 0.0) or 0.0) * multiplier,
                'magnesium': (item.get('magnesium_val', 0.0) or 0.0) * multiplier,
                'serving_multiplier': multiplier,
                'reason': ', '.join(reasons)
            })
            daily_protein += item.get('protein_val', 0.0) * multiplier
            daily_cal += item.get('calories_val', 0.0) * multiplier
            used_ids.add(item.get('_meal_planner_id'))
        
        plan[meal] = out_items
    
    # If snacks enabled, allocate remaining calories to a snack slot
    if include_snacks:
        snack_cal_budget = target_cal * 0.1 - daily_cal  # 10% snack budget
        if snack_cal_budget > 50:  # Only if significant budget remains
            available = [c for c in candidates if c.get('_meal_planner_id') not in used_ids]
            if available:
                ranked = rank_candidates(available)
                snack_items = _knapsack_select_meal(
                    ranked[:20],  # Top 20 only for snacks
                    capacity_cal=snack_cal_budget,
                    max_items=2,
                    protein_goal=5.0,
                    beta=0.3,
                    calories_weight=0.15
                )
                snack_out = []
                for item in snack_items:
                    # Compute reason tags for snacks
                    snack_reasons = ['snack']
                    if similar_user_likes and item.get('id') in similar_user_likes:
                        snack_reasons.append('similar_to_users_like_you')
                    
                    snack_out.append({
                        'id': item.get('id'),
                        'food': item.get('food') or item.get('name'),
                        'calories': item.get('calories_val', 0.0),
                        'protein': item.get('protein_val', 0.0),
                        'fiber': item.get('fiber_val', 0.0),
                        'magnesium': item.get('magnesium_val', 0.0),
                        'serving_multiplier': 1.0,
                        'reason': ', '.join(snack_reasons)
                    })
                    daily_protein += item.get('protein_val', 0.0)
                    daily_cal += item.get('calories_val', 0.0)
                plan['Snacks'] = snack_out
    
    return plan
