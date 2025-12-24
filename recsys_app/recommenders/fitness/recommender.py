"""Fitness recommendation model."""
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ..base import BaseRecommender
from recsys_app.core.utils import calculate_tdee

class FitnessRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.user_preprocessor = StandardScaler()
        self.activity_preprocessor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.user_feature_dim = None
        self.activity_feature_dim = None
        
    def build_model(self, user_dim: int, item_dim: int):
        """Build fitness recommendation model."""
        user_input = keras.layers.Input(shape=(user_dim,))
        activity_input = keras.layers.Input(shape=(item_dim,))
        user_embedding = keras.layers.Dense(64, activation='relu')(user_input)
        activity_embedding = keras.layers.Dense(64, activation='relu')(activity_input)
        interaction = keras.layers.Concatenate()([user_embedding, activity_embedding])
        x = keras.layers.Dense(128, activation='relu')(interaction)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        fitness_recommendation = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=[user_input, activity_input], outputs=fitness_recommendation)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
        self.model = model
        
    def preprocess_data(self, user_data: pd.DataFrame, activity_data: pd.DataFrame):
        """Preprocess user and activity data."""
        user_cols = ['age', 'weight', 'height', 'tdee']
        activity_cols = ["level", "bodypart", "equipment", "type"]
        # Ensure tdee exists
        ud = user_data.copy()
        if 'tdee' not in ud.columns:
            ud['tdee'] = ud.apply(lambda r: calculate_tdee(r.get('weight', 0), r.get('height', 0), r.get('age', 30), r.get('gender', 'M'), r.get('activity_level', 'medium')), axis=1)
        for col in user_cols:
            if col not in ud.columns:
                ud[col] = 0

        self.user_preprocessor.fit(ud[user_cols])
        user_features = self.user_preprocessor.transform(ud[user_cols])
        
        ad = activity_data.copy()
        for col in activity_cols:
            if col not in ad.columns:
                ad[col] = ""
        self.activity_preprocessor.fit(ad[activity_cols])
        activity_features = self.activity_preprocessor.transform(ad[activity_cols])
        
        self.user_feature_dim = user_features.shape[1]
        self.activity_feature_dim = activity_features.shape[1]
        self.database = activity_data
        
        if self.model is None:
            self.build_model(self.user_feature_dim, self.activity_feature_dim)
            
        return user_features, activity_features
        
    def train(self, user_data: pd.DataFrame, activity_data: pd.DataFrame, interactions: np.ndarray):
        """Train the fitness recommendation model."""
        user_features, activity_features = self.preprocess_data(user_data, activity_data)
        user_features_repeated = np.repeat(user_features, len(activity_features), axis=0)
        activity_features_tiled = np.tile(activity_features, (len(user_features), 1))
        interactions_flattened = interactions.flatten()
        self.model.fit(
            [user_features_repeated, activity_features_tiled],
            interactions_flattened,
            epochs=10,
            batch_size=32,
            verbose=1
        )
        # Store training-time interaction info for collaborative signals
        try:
            self.interactions_matrix = interactions.copy()
            # Simple popularity signal (number of positive interactions per activity)
            pop = interactions.sum(axis=0)
            # Avoid division by zero
            if pop.max() > 0:
                self.activity_popularity = pop / float(pop.max())
            else:
                self.activity_popularity = np.zeros_like(pop, dtype=float)
        except Exception:
            self.interactions_matrix = None
            self.activity_popularity = None
        
    def predict_scores(self, user: dict, activity_data: pd.DataFrame) -> np.ndarray:
        """Predict scores for a user and fitness items."""
        user_df = pd.DataFrame([{  # build safe user frame
            'age': user.get('age', 30),
            'weight': user.get('weight', 70.0),
            'height': user.get('height', 170.0),
            'tdee': user.get('tdee', calculate_tdee(user.get('weight', 70.0), user.get('height', 170.0), user.get('age', 30), user.get('gender', 'M'), user.get('activity_level', 'medium')))
        }])
        # Ensure columns exist
        for col in ['age','weight','height','tdee']:
            if col not in user_df.columns:
                user_df[col] = 0
        user_features = self.user_preprocessor.transform(user_df)
        activity_features = self.activity_preprocessor.transform(
            activity_data[['level', 'bodypart', 'equipment', 'type']]
        )
        user_features_repeated = np.repeat(user_features, len(activity_features), axis=0)
        scores = self.model.predict([user_features_repeated, activity_features], batch_size=128, verbose=0)
        return scores.flatten()
        
    def generate_recommendations(self, user: dict, top_k: int = 5, mode: str = "hybrid") -> pd.DataFrame:
        """Generate fitness recommendations for a user.
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
        for col in ['age', 'weight', 'height', 'tdee']:
            if col not in user_df.columns:
                user_df[col] = 0

        user_features = self.user_preprocessor.transform(user_df[['age', 'weight', 'height', 'tdee']])
        ad = self.database.copy()
        activity_features = self.activity_preprocessor.transform(
            ad[['level', 'bodypart', 'equipment', 'type']]
        )
        user_features_repeated = np.repeat(user_features, len(activity_features), axis=0)
        scores = self.model.predict([user_features_repeated, activity_features], batch_size=128, verbose=0).flatten()
        df = self.database.copy()

        # --- Collaborative: recommend by activity popularity ---
        if mode == "collaborative":
            if hasattr(self, "activity_popularity") and self.activity_popularity is not None:
                pop_scores = self.activity_popularity
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
        sim_scores = np.zeros(len(scores))
        sim_neg = np.zeros(len(scores))
        try:
            pos_items = None
            neg_items = None
            if getattr(self, 'user_positive_item_ids', None):
                try:
                    id_list = list(self.database['id'].tolist()) if 'id' in self.database.columns else list(range(len(self.database)))
                    pos_items = [id_list.index(pid) for pid in self.user_positive_item_ids if pid in id_list]
                except Exception:
                    pos_items = None
            if getattr(self, 'user_negative_item_ids', None):
                try:
                    id_list = list(self.database['id'].tolist()) if 'id' in self.database.columns else list(range(len(self.database)))
                    neg_items = [id_list.index(pid) for pid in self.user_negative_item_ids if pid in id_list]
                except Exception:
                    neg_items = None
            if pos_items is None and hasattr(self, 'interactions_matrix') and self.interactions_matrix is not None:
                if self.interactions_matrix.shape[0] > 0:
                    pos_items = np.where(self.interactions_matrix[0] > 0)[0]
            if pos_items is not None and len(pos_items) > 0:
                item_vecs = activity_features
                norms = np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-8
                item_vecs_norm = item_vecs / norms
                for i in range(len(item_vecs_norm)):
                    sims = item_vecs_norm[pos_items] @ item_vecs_norm[i]
                    sim_scores[i] = np.max(sims) if len(sims) > 0 else 0.0
            if neg_items is not None and len(neg_items) > 0:
                item_vecs = activity_features
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