"""Simple evaluation script to compare hybrid nutrition recommendations
with default vs tuned parameters for a few sample users.

Run from repo root with:
    python tools/eval_nutrition_tuning.py

It prints top-k nutrition item names and hybrid scores for each user and
for each parameter set.
"""
import pprint
from recsys_app.database.session import SessionLocal
from recsys_app.database.models import User, NutritionItem
import pandas as pd
from recsys_app.models.nutrition.utils import hybrid_nutrition_recommendations

TOP_K = 5

# Two parameter sets: "default" mirrors current defaults, "tuned" is more protein-focused
PARAM_SETS = {
    'default': {
        'alpha': 0.5,
        'prot_boost_weight': 0.30,
        'size_penalty_weight': 0.35,
        'size_threshold_frac': 0.4
    },
    'tuned_protein_focus': {
        'alpha': 0.3,  # less collaborative, more content-driven
        'prot_boost_weight': 0.45,
        'size_penalty_weight': 0.25,
        'size_threshold_frac': 0.30
    }
}

pp = pprint.PrettyPrinter(indent=2)


def fetch_data():
    db = SessionLocal()
    try:
        users = db.query(User).limit(3).all()
        nutrition_items = db.query(NutritionItem).all()
        n_df = pd.DataFrame([i.__dict__ for i in nutrition_items])
        return users, n_df
    finally:
        db.close()


def run_eval():
    users, n_df = fetch_data()
    if n_df.empty:
        print("No nutrition items found in DB — run init_sample_data or seed the DB first.")
        return

    for u in users:
        print('\n' + '='*60)
        print(f"User {u.id} (age={u.age}, weight={u.weight}, height={u.height}, gender={u.gender}, goals={u.health_goals})")
        user_dict = {
            'age': u.age,
            'weight': u.weight,
            'height': u.height,
            'gender': u.gender,
            'activity_level': u.activity_level,
            'health_goals': u.health_goals,
            'sleep_good': u.sleep_good
        }
        # Compute nutrition_targets similar to route logic
        try:
            from recsys_app.routes.recommendations import calculate_tdee as _calc_tdee
        except Exception:
            from recsys_app.core.utils import calculate_tdee as _calc_tdee
        try:
            w = float(u.weight)
            h = float(u.height)
            a = float(u.age)
        except Exception:
            w, h, a = 70.0, 170.0, 30.0
        try:
            if (u.health_goals or '').upper().startswith('MG'):
                protein_goal = w * 1.6
            else:
                protein_goal = w * 1.2
        except Exception:
            protein_goal = w * 1.2
        try:
            tdee = getattr(u, 'tdee', None) or _calc_tdee(w, h, a, u.gender, u.activity_level)
        except Exception:
            tdee = None

        nutrition_targets = {
            'calories': float(tdee) if tdee is not None else None,
            'protein_g': float(protein_goal),
            'carbs_g': None,
            'fat_g': None,
            'bmr': None
        }

        for pname, params in PARAM_SETS.items():
            print('\n-- Param set:', pname)
            recs = hybrid_nutrition_recommendations(
                user_dict,
                n_df,
                recommender=None,
                top_k=TOP_K,
                alpha=params['alpha'],
                nutrition_targets=nutrition_targets,
                prot_boost_weight=params['prot_boost_weight'],
                size_penalty_weight=params['size_penalty_weight'],
                size_threshold_frac=params['size_threshold_frac']
            )
            # Print top items and their hybrid_score
            for i, row in recs.head(TOP_K).iterrows():
                print(f"{int(row.get('id', -1))} | {row.get('food', '<no-name>')[:60]:60} | cal={row.get('calories')} | prot={row.get('protein')} | score={row.get('hybrid_score'):.4f}")


if __name__ == '__main__':
    run_eval()
