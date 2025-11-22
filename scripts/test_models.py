"""Quick script to test the recommender models locally.

This script:
 - ensures sample data/tables exist (calls init_sample_data)
 - loads users, nutrition items and fitness items from the app DB
 - creates small simulated interaction matrices and trains the recommenders
 - prints top-k recommendations for the first user

Run from project root:
    python scripts/test_models.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from recsys_app.database.session import SessionLocal, init_sample_data
from recsys_app.database.models import User, NutritionItem, FitnessItem
from recsys_app.models.nutrition.recommender import NutritionRecommender
from recsys_app.models.fitness.recommender import FitnessRecommender


def df_from_query(rows):
    return pd.DataFrame([{k: v for k, v in r.__dict__.items() if not k.startswith('_')} for r in rows])


def main(user_id: int = None, top_k: int = 5):
    init_sample_data()
    db = SessionLocal()
    try:
        users = db.query(User).all()
        nutrition = db.query(NutritionItem).all()
        fitness = db.query(FitnessItem).all()

        if not users:
            print("No users found in DB. Run scripts/seed_db.py first.")
            return

        users_df = df_from_query(users)
        nutrition_df = df_from_query(nutrition)
        fitness_df = df_from_query(fitness)

        # choose a user
        if user_id is None:
            user = users[0]
        else:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return

        user_dict = {k: v for k, v in user.__dict__.items() if not k.startswith('_')}

        # Create recommenders
        nutr_rec = NutritionRecommender()
        fit_rec = FitnessRecommender()

        # Prepare small interactions (simulate random prefs)
        U = len(users_df)
        I_n = len(nutrition_df)
        I_f = len(fitness_df)

        if U == 0 or I_n == 0:
            print("Insufficient data to train.")
            return

        np.random.seed(42)
        nutr_interactions = np.random.randint(0, 2, size=(U, I_n))
        fit_interactions = np.random.randint(0, 2, size=(U, I_f)) if I_f > 0 else np.zeros((U, max(1, I_f)))

        print("Training nutrition recommender (quick demo)...")
        try:
            nutr_rec.train(users_df, nutrition_df, nutr_interactions)
        except Exception as e:
            print("Nutrition training failed (falling back to content-based):", e)

        print("Training fitness recommender (quick demo)...")
        try:
            fit_rec.train(users_df, fitness_df, fit_interactions)
        except Exception as e:
            print("Fitness training failed (falling back to simple scoring):", e)

        # Get recommendations
        print(f"\nTop {top_k} Nutrition recommendations for user id={user.id}:")
        try:
            nutr_scores = nutr_rec.predict_scores(user_dict, nutrition_df)
            top_idx = nutr_scores.argsort()[-top_k:][::-1]
            for i in top_idx:
                row = nutrition_df.iloc[i]
                print(f" - {row.get('food','<no-name>')}  | calories={row.get('calories')} protein={row.get('protein')}")
        except Exception as e:
            print("Nutrition predict failed, showing top protein foods:")
            top = nutrition_df.sort_values('protein', ascending=False).head(top_k)
            for _, row in top.iterrows():
                print(f" - {row.get('food','<no-name>')} | protein={row.get('protein')}")

        print(f"\nTop {top_k} Fitness recommendations for user id={user.id}:")
        try:
            fit_scores = fit_rec.predict_scores(user_dict, fitness_df)
            top_idx = fit_scores.argsort()[-top_k:][::-1]
            for i in top_idx:
                row = fitness_df.iloc[i]
                print(f" - {row.get('name','<no-name>')} | level={row.get('level')} bodypart={row.get('bodypart')}")
        except Exception as e:
            print("Fitness predict failed, showing simple level-matched activities:")
            lv = user_dict.get('activity_level')
            matched = fitness_df[fitness_df['level'] == lv].head(top_k)
            for _, row in matched.iterrows():
                print(f" - {row.get('name','<no-name>')} | level={row.get('level')}")

    finally:
        db.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--user', type=int, help='User id to recommend for')
    p.add_argument('--top', type=int, default=5, help='Top-k')
    args = p.parse_args()
    main(user_id=args.user, top_k=args.top)
