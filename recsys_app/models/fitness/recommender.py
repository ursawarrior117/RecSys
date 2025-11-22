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
            batch_size=32
        )
        
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
        scores = self.model.predict([user_features_repeated, activity_features], batch_size=128)
        return scores.flatten()
        
    def generate_recommendations(self, user: dict, top_k: int = 5) -> pd.DataFrame:
        """Generate fitness recommendations for a user."""
        scores = self.predict_scores(user, self.database)
        top_indices = scores.argsort()[-top_k:][::-1]
        return self.database.iloc[top_indices]