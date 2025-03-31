import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

class NutritionRecommender:
    def __init__(self):
        self.user_preprocessor = StandardScaler()
        self.nutrition_preprocessor = StandardScaler()
        self.health_goal_encoder = MultiLabelBinarizer()
        self.user_feature_dim = None
        self.food_feature_dim = None
        self.nutrition_database = None
        self.nutrition_model = None

    def _build_nutrition_model(self):
        user_input = keras.layers.Input(shape=(self.user_feature_dim,))
        food_input = keras.layers.Input(shape=(self.food_feature_dim,))
        user_embedding = keras.layers.Dense(64, activation='relu')(user_input)
        food_embedding = keras.layers.Dense(64, activation='relu')(food_input)
        interaction = keras.layers.Concatenate()([user_embedding, food_embedding])
        x = keras.layers.Dense(128, activation='relu')(interaction)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        nutrition_recommendation = keras.layers.Dense(1, activation='sigmoid', name='nutrition_recommendation')(x)
        nutritional_alignment = keras.layers.Dense(1, activation='linear', name='nutritional_alignment')(x)
        model = keras.Model(inputs=[user_input, food_input], outputs=[nutrition_recommendation, nutritional_alignment])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'nutrition_recommendation': 'binary_crossentropy',
                'nutritional_alignment': 'mean_squared_error'
            }
        )
        return model

    def preprocess_data(self, user_data, nutrition_data):
        user_features = self.user_preprocessor.fit_transform(user_data[['age', 'weight', 'height']])
        health_goal_encoded = self.health_goal_encoder.fit_transform(user_data['health_goals'])
        processed_user_features = np.hstack([user_features, health_goal_encoded])
        self.user_feature_dim = processed_user_features.shape[1]
        processed_nutrition_features = self.nutrition_preprocessor.fit_transform(nutrition_data[['calories', 'protein', 'carbs', 'fat']])
        self.food_feature_dim = processed_nutrition_features.shape[1]
        self.nutrition_database = nutrition_data
        self.nutrition_model = self._build_nutrition_model()
        return processed_user_features, processed_nutrition_features

    def train(self, user_data, nutrition_data, nutrition_interactions):
        processed_users, processed_nutrition = self.preprocess_data(user_data, nutrition_data)
        user_nutrition_pairs = np.repeat(processed_users, len(processed_nutrition), axis=0)
        nutrition_pairs = np.tile(processed_nutrition, (len(processed_users), 1))
        self.nutrition_model.fit(
            [user_nutrition_pairs, nutrition_pairs],
            {
                'nutrition_recommendation': nutrition_interactions.flatten(),
                'nutritional_alignment': np.random.rand(len(user_nutrition_pairs), 1)
            },
            epochs=10,
            batch_size=32,
            verbose=1
        )

    def generate_recommendations(self, user, top_k=3):
        processed_user, processed_nutrition = self.preprocess_data(pd.DataFrame([user]), self.nutrition_database)
        user_nutrition_pairs = np.repeat(processed_user, len(processed_nutrition), axis=0)
        nutrition_recommendations, nutritional_scores = self.nutrition_model.predict([user_nutrition_pairs, processed_nutrition])
        sorted_indices = np.argsort(nutrition_recommendations.flatten())[::-1][:top_k]
        return [
            {
                'food_id': self.nutrition_database.iloc[i]['food_id'],
                'name': self.nutrition_database.iloc[i]['name'],
                'recommendation_score': nutrition_recommendations[i][0],
                'nutritional_alignment': nutritional_scores[i][0]
            }
            for i in sorted_indices
        ]