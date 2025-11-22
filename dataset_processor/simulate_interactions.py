import numpy as np
import pandas as pd
from datetime import datetime
from math import exp
from interaction_logger import load_interactions

DECAY_DAYS = 30.0  # time-decay window in days for popularity

def _time_decay_weight(ts, now):
    days = (now - ts).total_seconds() / 86400.0
    return exp(-days / DECAY_DAYS)

def _compute_popularity(items_df):
    """
    Deterministic popularity score per item based on logged interactions with time decay.
    Returns Series indexed by item_id (or items_df index) with normalized [0,1] scores.
    """
    logs = load_interactions()
    if logs.empty:
        # fallback to uniform popularity
        pop = pd.Series(1.0, index=items_df.index)
    else:
        now = datetime.utcnow()
        # ensure ts column exists and item_id integer
        if "ts" not in logs.columns:
            logs["ts"] = now
        logs["weight"] = logs["ts"].apply(lambda t: _time_decay_weight(t, now))
        agg = logs.groupby("item_id")["weight"].sum()
        # align with items_df index (item_id should match items_df.item_id if present)
        if "item_id" in items_df.columns:
            idx = items_df["item_id"].astype(int)
            pop = pd.Series(0.0, index=items_df.index)
            for i,row_idx in enumerate(idx):
                pop.iloc[i] = float(agg.get(int(row_idx), 0.0))
        else:
            # if no item_id column, assume aligned index
            pop = pd.Series(0.0, index=items_df.index)
            for iid in agg.index:
                if iid in pop.index:
                    pop.loc[iid] = float(agg.loc[iid])
    # normalize to 0..1
    if pop.max() == pop.min():
        return (pop - pop.min()).fillna(0.0)
    return (pop - pop.min()) / (pop.max() - pop.min())

def _normalize_series(s):
    if s.isnull().all():
        return s.fillna(0.0)
    if s.max() == s.min():
        return s.fillna(0.0)
    return (s - s.min()) / (s.max() - s.min())

def _nutrition_content_score(user_row, items_df):
    # Prefer high protein for MG, low calories for WL, neutral otherwise
    goals = str(user_row.get("health_goals", "")).upper()
    prot = items_df.get("protein", pd.Series(0.0, index=items_df.index)).astype(float)
    cal = items_df.get("calories", pd.Series(0.0, index=items_df.index)).astype(float)
    prot_n = _normalize_series(prot)
    cal_n = _normalize_series(cal)
    if goals == "MG":
        score = 0.7 * prot_n + 0.3 * (1 - cal_n)
    elif goals == "WL" or goals == "WL (WEIGHT LOSS)" or goals == "WEIGHT LOSS":
        score = 0.2 * prot_n + 0.8 * (1 - cal_n)
    else:
        score = 0.5 * prot_n + 0.5 * (1 - cal_n)
    return _normalize_series(score)

def _fitness_content_score(user_row, items_df):
    # Use difficulty/level where lower difficulty preferred for low activity, higher for high activity
    act = str(user_row.get("activity_level", "medium")).lower()
    level = None
    for col in ["level", "difficulty", "intensity"]:
        if col in items_df.columns:
            level = items_df[col]
            break
    if level is None:
        # fallback to zeros
        level_n = pd.Series(0.5, index=items_df.index)
    else:
        level_n = _normalize_series(pd.to_numeric(level, errors="coerce").fillna(level.mean()))
    if act == "low":
        score = 1 - level_n
    elif act == "high":
        score = level_n
    else:
        score = 1 - (abs(level_n - 0.5) * 0.5)
    return _normalize_series(score)

def simulate_interactions(user_df, items_df, item_type="nutrition", top_k_per_user=None):
    """
    Deterministically simulate interactions for a given set of users and items.
    Returns a 2D list / numpy array (n_users x n_items) with 1.0 for selected simulated interactions, 0.0 otherwise.
    This output is deterministic and driven by logged interactions (popularity) and content heuristics.
    """
    if items_df is None or items_df.empty or user_df is None or user_df.empty:
        # return empty matrix with proper shape
        n_users = 0 if user_df is None else len(user_df)
        n_items = 0 if items_df is None else len(items_df)
        return np.zeros((n_users, n_items), dtype=float).tolist()

    pop = _compute_popularity(items_df)
    n_items = len(items_df)
    n_users = len(user_df)
    if top_k_per_user is None:
        top_k = max(3, n_items // 40)  # deterministic default
    else:
        top_k = int(top_k_per_user)

    result = np.zeros((n_users, n_items), dtype=float)

    # precompute per-item content features depending on type
    if item_type.lower() == "nutrition":
        # ensure columns names normalized
        content_matrix = None
        # compute content score once per user because it depends on user goals
    else:
        # fitness
        pass

    for ui, (_, urow) in enumerate(user_df.iterrows()):
        if item_type.lower() == "nutrition":
            content_score = _nutrition_content_score(urow, items_df)
        else:
            content_score = _fitness_content_score(urow, items_df)

        # combine popularity and content deterministically
        # weights based on sleep_good and activity_level (small influence)
        sleep = int(urow.get("sleep_good", 1))
        alpha = 0.6 if sleep == 1 else 0.5  # content weight
        combined = alpha * content_score.values + (1 - alpha) * pop.values
        # select top_k deterministic items (stable ordering by score then item index)
        order = np.lexsort((np.arange(n_items), combined))  # lexsort yields ascending; use last entries
        top_idx = order[::-1][:top_k]
        result[ui, top_idx] = 1.0

    return result.tolist()