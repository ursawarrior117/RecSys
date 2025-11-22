"""Training service that trains recommenders and persists models."""
from recsys_app.database.session import SessionLocal, init_sample_data
from recsys_app.database.models import User, NutritionItem, FitnessItem
from recsys_app.models.nutrition.recommender import NutritionRecommender
from recsys_app.models.fitness.recommender import FitnessRecommender
from recsys_app import model_io
import pandas as pd
import numpy as np


def df_from_query(rows):
    return pd.DataFrame([{k: v for k, v in r.__dict__.items() if not k.startswith('_')} for r in rows])


def train_and_persist_models(simulate_interactions: bool = True):
    """Train nutrition and fitness recommenders on available DB data and save models.

    If no interactions table exists, this function will simulate interactions when
    `simulate_interactions` is True.
    """
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

        if nutr_interactions is not None:
            nutr_rec.train(users_df, nutrition_df, nutr_interactions)
        if fit_interactions is not None and not fitness_df.empty:
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

        return {"status": "ok", "models": model_io.list_models()}

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
