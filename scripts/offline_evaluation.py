"""Offline evaluation script for the recommender system.

This script performs offline evaluation using train/test splits on simulated or real interaction data.
It computes standard metrics: Precision@K, Recall@K, NDCG@K, MRR for both nutrition and fitness domains.

Run from project root:
    python scripts/offline_evaluation.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from recsys_app.database.session import SessionLocal, init_sample_data
from recsys_app.database.models import User, NutritionItem, FitnessItem, Interaction
from recsys_app.models.nutrition.recommender import NutritionRecommender
from recsys_app.models.fitness.recommender import FitnessRecommender


def df_from_query(rows):
    return pd.DataFrame([{k: v for k, v in r.__dict__.items() if not k.startswith('_')} for r in rows])


def precision_at_k(recommended_ids, relevant_set, k):
    if k <= 0 or not recommended_ids:
        return 0.0
    hits = sum(1 for rid in recommended_ids[:k] if rid in relevant_set)
    return hits / float(k)


def recall_at_k(recommended_ids, relevant_set, k):
    if len(relevant_set) == 0:
        return 0.0
    hits = sum(1 for rid in recommended_ids[:k] if rid in relevant_set)
    return hits / float(len(relevant_set))


def ndcg_at_k(recommended_ids, relevant_set, k):
    dcg = 0.0
    for i, rid in enumerate(recommended_ids[:k]):
        rel = 1.0 if rid in relevant_set else 0.0
        dcg += rel / np.log2(i + 2)
    ideal_rels = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(1, ideal_rels + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended_ids, relevant_set):
    for rank, rid in enumerate(recommended_ids, 1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_recommender(recommender, users_df, items_df, train_mat, test_mat, item_ids, k=10):
    """Evaluate a recommender on train/test matrices."""
    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []

    for u in range(len(users_df)):
        user_row = users_df.iloc[u].to_dict()
        try:
            # Get recommendations
            recs = recommender.generate_recommendations(user_row, top_k=k)
            rec_ids = [int(x) for x in recs['id'].tolist() if 'id' in recs.columns]
        except Exception:
            continue

        # Get relevant items from test set
        rel_idxs = np.where(test_mat[u] > 0)[0]
        relevant_ids = set(item_ids[i] for i in rel_idxs)
        if len(relevant_ids) == 0:
            continue

        precisions.append(precision_at_k(rec_ids, relevant_ids, k))
        recalls.append(recall_at_k(rec_ids, relevant_ids, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant_ids, k))
        mrrs.append(mrr(rec_ids, relevant_ids))

    metrics = {
        f'precision@{k}': np.mean(precisions) if precisions else 0.0,
        f'recall@{k}': np.mean(recalls) if recalls else 0.0,
        f'ndcg@{k}': np.mean(ndcgs) if ndcgs else 0.0,
        'mrr': np.mean(mrrs) if mrrs else 0.0,
        'evaluated_users': len(precisions)
    }
    return metrics


def main(k=10, test_size=0.2):
    init_sample_data()
    db = SessionLocal()
    try:
        users = db.query(User).all()
        nutrition_items = db.query(NutritionItem).all()
        fitness_items = db.query(FitnessItem).all()
        interactions = db.query(Interaction).all()

        users_df = df_from_query(users)
        nutrition_df = df_from_query(nutrition_items)
        fitness_df = df_from_query(fitness_items)

        if users_df.empty or nutrition_df.empty:
            print("Insufficient data for evaluation.")
            return

        print(f"Loaded {len(users_df)} users, {len(nutrition_df)} nutrition items, {len(fitness_df)} fitness items")
        print(f"Interactions in DB: {len(interactions)}")

        # Build interaction matrices
        user_ids = users_df['id'].tolist()
        nut_ids = nutrition_df['id'].tolist()
        fit_ids = fitness_df['id'].tolist()

        U = len(user_ids)
        I_n = len(nut_ids)
        I_f = len(fit_ids)

        nutr_mat = np.zeros((U, I_n), dtype=int)
        fit_mat = np.zeros((U, I_f), dtype=int) if I_f > 0 else None

        id_to_u = {uid: idx for idx, uid in enumerate(user_ids)}
        id_to_n = {nid: idx for idx, nid in enumerate(nut_ids)}
        id_to_f = {fid: idx for idx, fid in enumerate(fit_ids)}

        # Fill from DB interactions
        for it in interactions:
            uidx = id_to_u.get(it.user_id)
            if uidx is None:
                continue
            if it.nutrition_item_id and it.nutrition_item_id in id_to_n:
                nidx = id_to_n[it.nutrition_item_id]
                nutr_mat[uidx, nidx] = 1 if (it.rating or 0) > 0 else 0
            if it.fitness_item_id and fit_mat is not None and it.fitness_item_id in id_to_f:
                fidx = id_to_f[it.fitness_item_id]
                fit_mat[uidx, fidx] = 1 if (it.rating or 0) > 0 else 0

        # If no interactions, simulate
        if np.sum(nutr_mat) == 0:
            print("No nutrition interactions found, simulating random...")
            np.random.seed(42)
            nutr_mat = np.random.randint(0, 2, size=(U, I_n))
        if fit_mat is not None and np.sum(fit_mat) == 0:
            print("No fitness interactions found, simulating random...")
            np.random.seed(43)
            fit_mat = np.random.randint(0, 2, size=(U, I_f))

        # Split into train/test
        np.random.seed(42)
        # Split users for train/test
        train_users, test_users = train_test_split(users_df, test_size=test_size, random_state=42)
        train_indices = train_users.index.tolist()
        test_indices = test_users.index.tolist()
        
        train_nutr = nutr_mat[train_indices]
        test_nutr = nutr_mat[test_indices]
        if fit_mat is not None:
            train_fit = fit_mat[train_indices]
            test_fit = fit_mat[test_indices]
        else:
            train_fit = test_fit = None

        # Train recommenders
        nutr_rec = NutritionRecommender()
        fit_rec = FitnessRecommender()

        print("Training nutrition recommender...")
        nutr_rec.train(train_users, nutrition_df, train_nutr)

        if fit_mat is not None:
            print("Training fitness recommender...")
            fit_rec.train(train_users, fitness_df, train_fit)

        # Evaluate
        print("\nEvaluating Nutrition Recommender:")
        nutr_metrics = evaluate_recommender(nutr_rec, test_users, nutrition_df, train_nutr, test_nutr, nut_ids, k)
        for key, value in nutr_metrics.items():
            print(f"{key}: {value:.4f}")

        if fit_mat is not None:
            print("\nEvaluating Fitness Recommender:")
            fit_metrics = evaluate_recommender(fit_rec, test_users, fitness_df, train_fit, test_fit, fit_ids, k)
            for key, value in fit_metrics.items():
                print(f"{key}: {value:.4f}")

        print("\nEvaluation complete. Use these metrics in your thesis report.")

    finally:
        db.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=10, help='K for metrics')
    p.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    args = p.parse_args()
    main(k=args.k, test_size=args.test_size)