"""Training service that trains recommenders and persists models."""
from recsys_app.database.session import SessionLocal, init_sample_data
from recsys_app.database.models import User, NutritionItem, FitnessItem
from recsys_app.recommenders.nutrition.recommender import NutritionRecommender
from recsys_app.recommenders.fitness.recommender import FitnessRecommender
from recsys_app import model_io
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import func


def df_from_query(rows):
    return pd.DataFrame([{k: v for k, v in r.__dict__.items() if not k.startswith('_')} for r in rows])


def train_and_persist_models(simulate_interactions: bool = True):
    """Train nutrition and fitness recommenders on available DB data and save models.

    If no interactions table exists, this function will simulate interactions when
    `simulate_interactions` is True.
    """
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    init_sample_data()
    db = SessionLocal()
    try:
        users = db.query(User).all()
        nutrition = db.query(NutritionItem).all()
        fitness = db.query(FitnessItem).all()

        users_df = df_from_query(users)
        nutrition_df = df_from_query(nutrition)
        fitness_df = df_from_query(fitness)

        if users_df.empty or nutrition_df.empty:
            raise RuntimeError("Not enough data to train models (users or nutrition empty)")

        # Build recommenders
        nutr_rec = NutritionRecommender()
        fit_rec = FitnessRecommender()

        # Determine interactions: prefer recorded interactions in DB
        try:
            from recsys_app.database.models import Interaction
            interactions = db.query(Interaction).all()
        except Exception:
            interactions = []

        # Build user and item id maps
        user_ids = users_df['id'].tolist()
        nut_ids = nutrition_df['id'].tolist()
        fit_ids = fitness_df['id'].tolist() if len(fitness_df) > 0 else []

        U = len(user_ids)
        I_n = len(nut_ids)
        I_f = len(fit_ids)

        nutr_interactions = None
        fit_interactions = None

        if interactions:
            # Build matrices from recorded interactions (rating>0 considered positive)
            nutr_interactions = np.zeros((U, I_n), dtype=int)
            fit_interactions = np.zeros((U, I_f), dtype=int) if I_f > 0 else None
            id_to_u = {uid: idx for idx, uid in enumerate(user_ids)}
            id_to_n = {nid: idx for idx, nid in enumerate(nut_ids)}
            id_to_f = {fid: idx for idx, fid in enumerate(fit_ids)}
            for it in interactions:
                try:
                    uidx = id_to_u.get(it.user_id)
                    if it.nutrition_item_id and it.nutrition_item_id in id_to_n:
                        nidx = id_to_n[it.nutrition_item_id]
                        nutr_interactions[uidx, nidx] = 1 if (it.rating or 0) > 0 else 0
                    if it.fitness_item_id and fit_interactions is not None and it.fitness_item_id in id_to_f:
                        fidx = id_to_f[it.fitness_item_id]
                        fit_interactions[uidx, fidx] = 1 if (it.rating or 0) > 0 else 0
                except Exception:
                    continue
        else:
            # Simulate interactions if enabled
            if simulate_interactions:
                np.random.seed(0)
                nutr_interactions = np.random.randint(0, 2, size=(U, I_n))
                fit_interactions = np.random.randint(0, 2, size=(U, I_f)) if I_f > 0 else None

        # If we have recorded interactions, perform a small train/test holdout
        eval_metrics = None
        if nutr_interactions is not None:
            # create train/test matrices: hold out one positive item per user when possible
            train_mat = nutr_interactions.copy()
            test_mat = np.zeros_like(nutr_interactions)
            rng = np.random.RandomState(42)
            users_for_eval = []
            id_to_n = {nid: idx for idx, nid in enumerate(nut_ids)}
            idx_to_itemid = {idx: nid for idx, nid in enumerate(nut_ids)}
            for u in range(U):
                pos = np.where(nutr_interactions[u] > 0)[0]
                if len(pos) >= 2:
                    hold = rng.choice(pos)
                    train_mat[u, hold] = 0
                    test_mat[u, hold] = 1
                    users_for_eval.append(u)

            # train on the train_mat
            print("Training nutrition recommender...")
            nutr_rec.train(users_df, nutrition_df, train_mat)

            # Evaluate recommendations using the held-out test items
            def precision_at_k(recommended_ids, relevant_set, k):
                if k <= 0:
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
                    denom = np.log2(i + 2)
                    dcg += (2**rel - 1) / denom
                ideal_rels = [1.0] * min(len(relevant_set), k)
                idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rels))
                return dcg / idcg if idcg > 0 else 0.0

            K = 10
            precisions = []
            recalls = []
            ndcgs = []
            for u in users_for_eval:
                user_row = users_df.iloc[u].to_dict()
                try:
                    recs = nutr_rec.generate_recommendations(user_row, top_k=K)
                except Exception:
                    continue
                # recommended ids (assumes 'id' column present)
                rec_ids = [int(x) for x in recs['id'].tolist() if 'id' in recs.columns]
                rel_idxs = np.where(test_mat[u] > 0)[0]
                relevant_ids = set(idx_to_itemid[i] for i in rel_idxs)
                if len(relevant_ids) == 0:
                    continue
                precisions.append(precision_at_k(rec_ids, relevant_ids, K))
                recalls.append(recall_at_k(rec_ids, relevant_ids, K))
                ndcgs.append(ndcg_at_k(rec_ids, relevant_ids, K))

            if len(precisions) > 0:
                eval_metrics = {
                    'precision_at_10': float(np.mean(precisions)),
                    'recall_at_10': float(np.mean(recalls)),
                    'ndcg_at_10': float(np.mean(ndcgs)),
                    'evaluated_users': int(len(precisions))
                }
            else:
                eval_metrics = {'precision_at_10': 0.0, 'recall_at_10': 0.0, 'ndcg_at_10': 0.0, 'evaluated_users': 0}
        else:
            # No recorded interactions -> train on simulated or provided interactions
            if nutr_interactions is not None:
                print("Training nutrition recommender...")
                nutr_rec.train(users_df, nutrition_df, nutr_interactions)

        if fit_interactions is not None and not fitness_df.empty:
            print("Training fitness recommender...")
            fit_rec.train(users_df, fitness_df, fit_interactions)

        # Persist models and preprocessors
        if nutr_rec.model is not None:
            model_io.save_keras_model(nutr_rec.model, "nutrition")
            try:
                model_io.save_preprocessor(nutr_rec.user_preprocessor, "nutrition_user")
                model_io.save_preprocessor(nutr_rec.item_preprocessor, "nutrition_item")
            except Exception:
                pass

        if fit_rec.model is not None:
            model_io.save_keras_model(fit_rec.model, "fitness")
            try:
                model_io.save_preprocessor(fit_rec.user_preprocessor, "fitness_user")
                model_io.save_preprocessor(fit_rec.activity_preprocessor, "fitness_activity")
            except Exception:
                pass

        result = {"status": "ok", "models": model_io.list_models()}
        if eval_metrics is not None:
            result['eval_metrics'] = eval_metrics
        return result

    finally:
        db.close()


def load_models():
    """Load persisted models and preprocessors and return recommender instances.

    Returns a dict: { 'nutrition': NutritionRecommender instance or None, 'fitness': FitnessRecommender or None }
    """
    nutr_rec = NutritionRecommender()
    fit_rec = FitnessRecommender()

    # Try loading Keras models and preprocessors
    try:
        m = model_io.load_keras_model('nutrition')
        if m is not None:
            nutr_rec.model = m
        up = model_io.load_preprocessor('nutrition_user')
        ip = model_io.load_preprocessor('nutrition_item')
        if up is not None:
            nutr_rec.user_preprocessor = up
        if ip is not None:
            nutr_rec.item_preprocessor = ip
    except Exception:
        nutr_rec = None

    try:
        m = model_io.load_keras_model('fitness')
        if m is not None:
            fit_rec.model = m
        up = model_io.load_preprocessor('fitness_user')
        ap = model_io.load_preprocessor('fitness_activity')
        if up is not None:
            fit_rec.user_preprocessor = up
        if ap is not None:
            fit_rec.activity_preprocessor = ap
    except Exception:
        fit_rec = None

    return {"nutrition": nutr_rec, "fitness": fit_rec}


def retrain_if_needed(min_new_interactions: int = 100, window_days: int = 7):
    """Check recent interactions and retrain models if the count exceeds threshold.

    Returns a dict with status and the inspected interaction count.
    """
    init_sample_data()
    db = SessionLocal()
    try:
        # If there's no Interaction model, bail out gracefully
        try:
            from recsys_app.database.models import Interaction
        except Exception:
            return {"status": "no_interaction_table", "count": 0}

        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=window_days)
        count = db.query(func.count(Interaction.id)).filter(Interaction.timestamp >= cutoff).scalar() or 0
        if count >= min_new_interactions:
            # Record training run start
            try:
                from recsys_app.database.models import TrainingRun
                trun = TrainingRun(model_type='both', started_at=datetime.datetime.utcnow(), status='running')
                db.add(trun)
                db.commit()
                db.refresh(trun)
            except Exception:
                trun = None

            # perform a full retrain using recorded interactions
            started = datetime.datetime.utcnow()
            try:
                result = train_and_persist_models(simulate_interactions=False)
                status = 'ok'
            except Exception as e:
                result = {"error": str(e)}
                status = 'failed'
            finished = datetime.datetime.utcnow()

            # Update training run record with results
            try:
                if trun is not None:
                    trun.finished_at = finished
                    trun.status = status
                    trun.metrics = str(result)
                    db.add(trun)
                    db.commit()
            except Exception:
                pass

            return {"status": "retrained", "count": int(count), "result": result}
        else:
            return {"status": "not_enough_interactions", "count": int(count)}
    finally:
        db.close()
