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
        self.model.fit([X_user, X_item], y, epochs=5, batch_size=32, verbose=0)
        
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
        
    def generate_recommendations(self, user: dict, top_k: int = 5) -> pd.DataFrame:
        """Generate nutrition recommendations for a user."""
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
        scores = self.model.predict([user_features_repeated, nutrition_features], batch_size=128, verbose=0)
        top_indices = scores.flatten().argsort()[-top_k:][::-1]
        return self.database.iloc[top_indices]