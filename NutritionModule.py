import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

class NutritionRecommender:
    def __init__(self):
        self.user_preprocessor = StandardScaler()
        self.nutrition_preprocessor = StandardScaler()
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
        model = keras.Model(inputs=[user_input, food_input], outputs=nutrition_recommendation)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
        return model

    def preprocess_data(self, user_data, nutrition_data):
        user_features = self.user_preprocessor.fit_transform(user_data[['age', 'weight', 'height']])
        required_columns = [
            "calories", "fat", "saturated_fats", "monounsaturated_fats", "polyunsaturated_fats",
            "carbohydrates", "sugars", "protein", "fiber", "cholesterol", "sodium"
        ]
        processed_nutrition_features = self.nutrition_preprocessor.fit_transform(nutrition_data[required_columns])
        self.user_feature_dim = user_features.shape[1]
        self.food_feature_dim = processed_nutrition_features.shape[1]
        self.nutrition_database = nutrition_data
        self.nutrition_model = self._build_nutrition_model()
        return user_features, processed_nutrition_features

    def train(self, user_data, nutrition_data, interactions):
        """
        Train the nutrition recommendation model.
        """
        # Preprocess data
        user_features, nutrition_features = self.preprocess_data(user_data, nutrition_data)

        # Create Cartesian product of users and nutrition items
        user_features_repeated = np.repeat(user_features, len(nutrition_features), axis=0)
        nutrition_features_tiled = np.tile(nutrition_features, (len(user_features), 1))

        # Flatten interactions to match the Cartesian product
        interactions_flattened = interactions.flatten()

        # Train the model
        self.nutrition_model.fit(
            [user_features_repeated, nutrition_features_tiled],
            interactions_flattened,
            epochs=10,
            batch_size=32
        )

    def generate_recommendations(self, user, top_k=5):
        """
        Generate nutrition recommendations for a user.
        """
        # Convert the user data to a DataFrame with a single row
        user_features = self.user_preprocessor.transform(
            pd.DataFrame([user[['age', 'weight', 'height']].values], columns=['age', 'weight', 'height'])
        )

        # Use the same required columns as in the preprocess_data method
        required_columns = [
            "calories", "fat", "saturated_fats", "monounsaturated_fats", "polyunsaturated_fats",
            "carbohydrates", "sugars", "protein", "fiber", "cholesterol", "sodium"
        ]

        # Transform nutrition data using the same columns
        nutrition_features = self.nutrition_preprocessor.transform(self.nutrition_database[required_columns])

        # Repeat user features to match the number of nutrition items
        user_features_repeated = np.repeat(user_features, len(nutrition_features), axis=0)

        # Predict scores for all nutrition items
        scores = self.nutrition_model.predict([user_features_repeated, nutrition_features], batch_size=128)

        # Get the indices of the top_k nutrition items
        top_indices = scores.flatten().argsort()[-top_k:][::-1]

        # Return the top_k nutrition items
        return self.nutrition_database.iloc[top_indices]