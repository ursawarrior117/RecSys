"""Evaluation script for collaborative, content-based, and hybrid recommenders."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from recsys_app.database.session import SessionLocal, init_sample_data
from recsys_app.database.models import User, NutritionItem, FitnessItem, Interaction
from recsys_app.recommenders.nutrition.recommender import NutritionRecommender
from recsys_app.recommenders.fitness.recommender import FitnessRecommender

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

def evaluate_recommender(recommender, users_df, items_df, train_mat, test_mat, item_ids, k=10, mode="hybrid"):
    precisions, recalls, ndcgs, mrrs = [], [], [], []
    for u in range(len(users_df)):
        user_row = users_df.iloc[u].to_dict()
        rel_idxs = np.where(test_mat[u] > 0)[0]
        relevant_ids = set(item_ids[i] for i in rel_idxs)
        if len(relevant_ids) == 0:
            print(f"[DEBUG] User {user_row.get('id','?')} has no test positives.")
            continue
        # Set user-specific positive and negative item ids for the recommender
        pos_item_ids = [item_ids[i] for i in np.where(train_mat[u] > 0)[0]]
        neg_item_ids = [item_ids[i] for i in np.where(train_mat[u] == 0)[0]]
        recommender.user_positive_item_ids = pos_item_ids
        recommender.user_negative_item_ids = neg_item_ids
        if u < 3:  # debug first 3 users
            print(f"[DEBUG] User {u} features: {user_row}")
            print(f"[DEBUG] User {u} pos_items: {pos_item_ids[:5]}...")  # first 5
        try:
            recs = recommender.generate_recommendations(user_row, top_k=k, mode=mode)
            rec_ids = [int(x) for x in recs['id'].tolist() if 'id' in recs.columns]
        except Exception as e:
            print(f"[DEBUG] Recommendation failed for user {user_row.get('id','?')}: {e}")
            continue
        overlap = set(rec_ids) & relevant_ids
        print(f"[DEBUG] User {user_row.get('id','?')} test positives: {relevant_ids}, recs: {rec_ids}, overlap: {overlap}")
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
    users = db.query(User).all()
    nutrition_items = db.query(NutritionItem).all()
    fitness_items = db.query(FitnessItem).all()
    interactions = db.query(Interaction).all()
    users_df = df_from_query(users)
    nutrition_df = df_from_query(nutrition_items)
    fitness_df = df_from_query(fitness_items)
    db.close()
    if users_df.empty or nutrition_df.empty:
        print("Insufficient data for evaluation.")
        return
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
    if np.sum(nutr_mat) == 0:
        print("No nutrition interactions found, simulating random...")
        np.random.seed(42)
        nutr_mat = np.random.randint(0, 2, size=(U, I_n))
    if fit_mat is not None and np.sum(fit_mat) == 0:
        print("No fitness interactions found, simulating random...")
        np.random.seed(43)
        fit_mat = np.random.randint(0, 2, size=(U, I_f))

    # Leave-one-out split: for each user, hold out one positive for test, rest for train
    rng = np.random.RandomState(42)
    train_nutr = nutr_mat.copy()
    test_nutr = np.zeros_like(nutr_mat)
    train_fit = fit_mat.copy() if fit_mat is not None else None
    test_fit = np.zeros_like(fit_mat) if fit_mat is not None else None
    for u in range(U):
        # Nutrition
        pos_nut = np.where(nutr_mat[u] > 0)[0]
        if len(pos_nut) >= 2:
            test_idx = rng.choice(pos_nut, size=1, replace=False)
            train_nutr[u, test_idx] = 0
            test_nutr[u, test_idx] = 1
        elif len(pos_nut) == 1:
            # If only one positive, use it for test
            test_nutr[u, pos_nut[0]] = 1
            train_nutr[u, pos_nut[0]] = 0
        # Fitness
        if fit_mat is not None:
            pos_fit = np.where(fit_mat[u] > 0)[0]
            if len(pos_fit) >= 2:
                test_idx = rng.choice(pos_fit, size=1, replace=False)
                train_fit[u, test_idx] = 0
                test_fit[u, test_idx] = 1
            elif len(pos_fit) == 1:
                test_fit[u, pos_fit[0]] = 1
                train_fit[u, pos_fit[0]] = 0

    train_users = users_df.copy()
    test_users = users_df.copy()

    # --- Trivial recommender baseline: always recommend user's known positives from train set ---
    class TrivialRecommender:
        def __init__(self, train_mat, item_ids, test_mat=None):
            self.train_mat = train_mat
            self.item_ids = item_ids
            self.test_mat = test_mat
        def generate_recommendations(self, user, top_k=10, mode=None):
            # Recommend the user's known positives from train set, always include test positive, pad with random if needed
            uidx = user_ids.index(user['id']) if user['id'] in user_ids else 0
            pos = np.where(self.train_mat[uidx] > 0)[0]
            recs = [self.item_ids[i] for i in pos]
            # Always include the test positive if not already present
            if self.test_mat is not None:
                test_pos = np.where(self.test_mat[uidx] > 0)[0]
                for t in test_pos:
                    tid = self.item_ids[t]
                    if tid not in recs:
                        recs.insert(0, tid)
            if len(recs) < top_k:
                # pad with random unseen
                unseen = [i for i in self.item_ids if i not in recs]
                np.random.shuffle(unseen)
                recs += unseen[:top_k-len(recs)]
            return pd.DataFrame({'id': recs[:top_k]})



    print("\nEvaluating Nutrition Trivial Recommender (train+test positives):")
    nutr_trivial = TrivialRecommender(train_nutr, nut_ids, test_mat=test_nutr)
    # Debug: print train/test split and recommendations for first 5 users
    for u in range(min(5, U)):
        user = users_df.iloc[u].to_dict()
        train_pos = [nut_ids[i] for i in np.where(train_nutr[u] > 0)[0]]
        test_pos = [nut_ids[i] for i in np.where(test_nutr[u] > 0)[0]]
        recs = nutr_trivial.generate_recommendations(user, top_k=k, mode=None)['id'].tolist()
        print(f"[DEBUG][User {user['id']}] train_pos: {train_pos}, test_pos: {test_pos}, recs: {recs}")
    nutr_metrics_trivial = evaluate_recommender(nutr_trivial, test_users, nutrition_df, train_nutr, test_nutr, nut_ids, k, mode=None)
    for key, value in nutr_metrics_trivial.items():
        print(f"[Trivial] {key}: {value:.4f}")

    print("\nEvaluating Fitness Trivial Recommender (train+test positives):")
    fit_trivial = TrivialRecommender(train_fit, fit_ids, test_mat=test_fit)
    fit_metrics_trivial = evaluate_recommender(fit_trivial, test_users, fitness_df, train_fit, test_fit, fit_ids, k, mode=None)
    for key, value in fit_metrics_trivial.items():
        print(f"[Trivial] {key}: {value:.4f}")

    nutr_rec = NutritionRecommender()
    fit_rec = FitnessRecommender()
    nutr_rec.train(train_users, nutrition_df, train_nutr)
    print("Training fitness recommender (hybrid)...")
    fit_rec.train(train_users, fitness_df, train_fit)
    print("\nEvaluating Nutrition Recommender (hybrid):")
    nutr_metrics_hybrid = evaluate_recommender(nutr_rec, test_users, nutrition_df, train_nutr, test_nutr, nut_ids, k, mode="hybrid")
    for key, value in nutr_metrics_hybrid.items():
        print(f"[Hybrid] {key}: {value:.4f}")
    print("\nEvaluating Nutrition Recommender (collaborative):")
    nutr_metrics_collab = evaluate_recommender(nutr_rec, test_users, nutrition_df, train_nutr, test_nutr, nut_ids, k, mode="collaborative")
    for key, value in nutr_metrics_collab.items():
        print(f"[Collaborative] {key}: {value:.4f}")
    print("\nEvaluating Nutrition Recommender (content):")
    nutr_metrics_content = evaluate_recommender(nutr_rec, test_users, nutrition_df, train_nutr, test_nutr, nut_ids, k, mode="content")
    for key, value in nutr_metrics_content.items():
        print(f"[Content] {key}: {value:.4f}")
    print("\nEvaluating Fitness Recommender (hybrid):")
    fit_metrics_hybrid = evaluate_recommender(fit_rec, test_users, fitness_df, train_fit, test_fit, fit_ids, k, mode="hybrid")
    for key, value in fit_metrics_hybrid.items():
        print(f"[Hybrid] {key}: {value:.4f}")
    print("\nEvaluating Fitness Recommender (collaborative):")
    fit_metrics_collab = evaluate_recommender(fit_rec, test_users, fitness_df, train_fit, test_fit, fit_ids, k, mode="collaborative")
    for key, value in fit_metrics_collab.items():
        print(f"[Collaborative] {key}: {value:.4f}")
    print("\nEvaluating Fitness Recommender (content):")
    fit_metrics_content = evaluate_recommender(fit_rec, test_users, fitness_df, train_fit, test_fit, fit_ids, k, mode="content")
    for key, value in fit_metrics_content.items():
        print(f"[Content] {key}: {value:.4f}")
    
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=10, help='K for metrics')
    p.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    args = p.parse_args()
    main(k=args.k, test_size=args.test_size)
