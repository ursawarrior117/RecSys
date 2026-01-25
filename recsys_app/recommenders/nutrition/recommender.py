"""Nutrition recommendation model."""
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from ..base import BaseRecommender
from recsys_app.core.utils import calculate_tdee

class NutritionRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.user_feature_cols = ['age', 'weight', 'height', 'tdee', 'sleep_good']
        self.item_feature_cols = [
            'calories', 'fat', 'protein', 'carbohydrates', 'fiber', 'sugars',
            'cholesterol', 'sodium', 'monounsaturated_fats', 'polyunsaturated_fats',
            'saturated_fats', 'zinc', 'calcium', 'magnesium', 'caffeine'
        ]
        self.user_preprocessor = StandardScaler()
        self.item_preprocessor = StandardScaler()
        
    def build_model(self, user_dim: int, item_dim: int):
        """Build nutrition recommendation model."""
        input_user = keras.Input(shape=(user_dim,))
        input_item = keras.Input(shape=(item_dim,))
        x = keras.layers.Concatenate()([input_user, input_item])
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=[input_user, input_item], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        
    def preprocess_data(self, user_data: pd.DataFrame, nutrition_data: pd.DataFrame):
        """Preprocess user and nutrition data."""
        nutrition_data = nutrition_data[
            (nutrition_data['protein'] < 100) &
            (nutrition_data['calories'] < 1500) &
            (nutrition_data['fat'] < 100)
        ]
        # Ensure required user columns exist (compute tdee if missing)
        ud = user_data.copy()
        if 'tdee' not in ud.columns:
            ud['tdee'] = ud.apply(lambda r: calculate_tdee(r.get('weight', 0), r.get('height', 0), r.get('age', 30), r.get('gender', 'M'), r.get('activity_level', 'medium')), axis=1)
        for col in self.user_feature_cols:
            if col not in ud.columns:
                ud[col] = 0

        # Ensure nutrition dataframe has item feature columns
        nd = nutrition_data.copy()
        for col in self.item_feature_cols:
            if col not in nd.columns:
                nd[col] = 0

        user_features = self.user_preprocessor.fit_transform(ud[self.user_feature_cols])
        item_features = self.item_preprocessor.fit_transform(nd[self.item_feature_cols])
        self.database = nutrition_data.copy()
        return user_features, item_features
        
    def train(self, user_data: pd.DataFrame, nutrition_data: pd.DataFrame, interactions: np.ndarray):
        """Train the nutrition recommendation model."""
        user_features, nutrition_features = self.preprocess_data(user_data, nutrition_data)
        if self.model is None:
            self.build_model(user_features.shape[1], nutrition_features.shape[1])
        X_user = np.repeat(user_features, nutrition_features.shape[0], axis=0)
        X_item = np.tile(nutrition_features, (user_features.shape[0], 1))
        y = interactions.flatten()
        self.model.fit([X_user, X_item], y, epochs=10, batch_size=32, verbose=1)
        # store training-time interaction info for simple collaborative signals
        try:
            self.interactions_matrix = interactions.copy()
            # simple popularity signal (number of positive interactions per item)
            pop = interactions.sum(axis=0)
            # avoid division by zero
            if pop.max() > 0:
                self.item_popularity = pop / float(pop.max())
            else:
                self.item_popularity = np.zeros_like(pop, dtype=float)
        except Exception:
            self.interactions_matrix = None
            self.item_popularity = None
        
    def predict_scores(self, user: dict, nutrition_data: pd.DataFrame) -> np.ndarray:
        """Predict scores for a user and nutrition items."""
        # Prepare user DataFrame, compute tdee if missing
        user_df = pd.DataFrame([user]).copy()
        if 'tdee' not in user_df.columns:
            user_df['tdee'] = user_df.apply(lambda r: calculate_tdee(r.get('weight', 0), r.get('height', 0), r.get('age', 30), r.get('gender', 'M'), r.get('activity_level', 'medium')), axis=1)
        for col in self.user_feature_cols:
            if col not in user_df.columns:
                user_df[col] = 0

        user_features = self.user_preprocessor.transform(
            pd.concat([user_df]*len(nutrition_data), ignore_index=True)
        )

        nd = nutrition_data.copy()
        for col in self.item_feature_cols:
            if col not in nd.columns:
                nd[col] = 0
        nutrition_features = self.item_preprocessor.transform(
            nd[self.item_feature_cols]
        )
        scores = self.model.predict([user_features, nutrition_features], verbose=0)
        return scores.flatten()
        
    def generate_recommendations(self, user: dict, top_k: int = 5, mode: str = "hybrid") -> pd.DataFrame:
        """Generate nutrition recommendations for a user.
        Args:
            user: user feature dict
            top_k: number of recommendations
            mode: 'hybrid', 'collaborative', or 'content'
        Returns:
            DataFrame of top_k recommended items
        """
        user_df = pd.DataFrame([user]).copy()
        if 'tdee' not in user_df.columns:
            user_df['tdee'] = user_df.apply(lambda r: calculate_tdee(r.get('weight', 0), r.get('height', 0), r.get('age', 30), r.get('gender', 'M'), r.get('activity_level', 'medium')), axis=1)
        for col in self.user_feature_cols:
            if col not in user_df.columns:
                user_df[col] = 0

        user_features = self.user_preprocessor.transform(user_df[self.user_feature_cols])
        nd = self.database.copy()
        for col in self.item_feature_cols:
            if col not in nd.columns:
                nd[col] = 0
        nutrition_features = self.item_preprocessor.transform(nd[self.item_feature_cols])
        user_features_repeated = np.repeat(user_features, len(nutrition_features), axis=0)
        scores = self.model.predict([user_features_repeated, nutrition_features], batch_size=128, verbose=0).flatten()
        df = self.database.copy()

        # --- Collaborative: recommend by item popularity ---
        if mode == "collaborative":
            if hasattr(self, "item_popularity") and self.item_popularity is not None:
                pop_scores = self.item_popularity
                top_indices = np.argsort(pop_scores)[-top_k:][::-1]
                return df.iloc[top_indices]
            else:
                # fallback to random if no popularity
                top_indices = np.random.choice(len(df), size=top_k, replace=False)
                return df.iloc[top_indices]

        # --- Content-based: recommend by model score only ---
        if mode == "content":
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return df.iloc[top_indices]

        # --- Hybrid: blend model and similarity ---
        # Item-based similarity: for the current user, find items they've interacted with (positively)
        sim_scores = np.zeros(len(scores))
        sim_neg = np.zeros(len(scores))
        try:
            pos_items = None
            neg_items = None
            if getattr(self, 'user_positive_item_ids', None):
                try:
                    id_list = list(self.database['id'].tolist())
                    pos_items = [id_list.index(pid) for pid in self.user_positive_item_ids if pid in id_list]
                except Exception:
                    pos_items = None
            if getattr(self, 'user_negative_item_ids', None):
                try:
                    id_list = list(self.database['id'].tolist())
                    neg_items = [id_list.index(pid) for pid in self.user_negative_item_ids if pid in id_list]
                except Exception:
                    neg_items = None
            if pos_items is None and hasattr(self, 'interactions_matrix') and self.interactions_matrix is not None:
                if self.interactions_matrix.shape[0] > 0:
                    pos_items = np.where(self.interactions_matrix[0] > 0)[0]
            if pos_items is not None and len(pos_items) > 0:
                item_vecs = nutrition_features
                norms = np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-8
                item_vecs_norm = item_vecs / norms
                for i in range(len(item_vecs_norm)):
                    sims = item_vecs_norm[pos_items] @ item_vecs_norm[i]
                    sim_scores[i] = np.max(sims) if len(sims) > 0 else 0.0
            if neg_items is not None and len(neg_items) > 0:
                item_vecs = nutrition_features
                norms = np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-8
                item_vecs_norm = item_vecs / norms
                for i in range(len(item_vecs_norm)):
                    sims = item_vecs_norm[neg_items] @ item_vecs_norm[i]
                    sim_neg[i] = np.max(sims) if len(sims) > 0 else 0.0
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
        except Exception:
            sim_scores = np.zeros(len(scores))
            sim_neg = np.zeros(len(scores))
        alpha = 0.1  # model score weight
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            norm_scores = (scores - smin) / (smax - smin)
        else:
            norm_scores = np.zeros_like(scores)
        negative_penalty = 0.8
        sim_effect = sim_scores - negative_penalty * sim_neg
        hybrid = alpha * norm_scores + (1.0 - alpha) * sim_effect
        try:
            df['hybrid_score'] = hybrid
        except Exception:
            df = df.reset_index(drop=True)
            df['hybrid_score'] = list(hybrid)
        top_indices = np.argsort(hybrid)[-top_k:][::-1]
        return df.iloc[top_indices]