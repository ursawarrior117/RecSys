import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataset_processor.fitness_processor import FitnessDatasetProcessor

class FitnessRecommender:
    def __init__(self):
        self.user_preprocessor = StandardScaler()
        self.activity_preprocessor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.activity_database = None
        self.fitness_model = None

    def _build_fitness_model(self):
        user_input = keras.layers.Input(shape=(self.user_feature_dim,))
        activity_input = keras.layers.Input(shape=(self.activity_feature_dim,))
        user_embedding = keras.layers.Dense(64, activation='relu')(user_input)
        activity_embedding = keras.layers.Dense(64, activation='relu')(activity_input)
        interaction = keras.layers.Concatenate()([user_embedding, activity_embedding])
        x = keras.layers.Dense(128, activation='relu')(interaction)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        fitness_recommendation = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=[user_input, activity_input], outputs=fitness_recommendation)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
        return model

    def preprocess_data(self, user_data, activity_data):
        self.user_preprocessor.fit(user_data[['age', 'weight', 'height', 'tdee']])
        user_features = self.user_preprocessor.transform(user_data[['age', 'weight', 'height', 'tdee']])
        required_columns = ["level", "bodypart", "equipment", "type"]
        self.activity_preprocessor.fit(activity_data[required_columns])
        activity_features = self.activity_preprocessor.transform(activity_data[required_columns])
        self.user_feature_dim = user_features.shape[1]
        self.activity_feature_dim = activity_features.shape[1]
        self.activity_database = activity_data
        self.fitness_model = self._build_fitness_model()
        return user_features, activity_features

    def train(self, user_data, activity_data, interactions):
        user_features, activity_features = self.preprocess_data(user_data, activity_data)
        user_features_repeated = np.repeat(user_features, len(activity_features), axis=0)
        activity_features_tiled = np.tile(activity_features, (len(user_features), 1))
        interactions_flattened = interactions.flatten()
        self.fitness_model.fit(
            [user_features_repeated, activity_features_tiled],
            interactions_flattened,
            epochs=10,
            batch_size=32
        )

    def generate_recommendations(self, user, top_k=5):
        user_df = pd.DataFrame([{
            'age': user['age'],
            'weight': user['weight'],
            'height': user['height'],
            'tdee': user['tdee']
        }])
        if not hasattr(self.user_preprocessor, 'mean_'):
            self.user_preprocessor.fit(user_df)
        user_features = self.user_preprocessor.transform(user_df)
        fitness_features = self.activity_preprocessor.transform(
            self.activity_database[['level', 'bodypart', 'equipment', 'type']]
        )
        user_features_repeated = np.repeat(user_features, len(fitness_features), axis=0)
        scores = self.fitness_model.predict([user_features_repeated, fitness_features], batch_size=128)
        top_indices = scores.flatten().argsort()[-top_k:][::-1]
        return self.activity_database.iloc[top_indices]

def main():
    # Load user data
    user_data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, 30, 35],
        "weight": [70, 60, 80],  # Weight in kilograms
        "height": [175, 160, 180],  # Height in centimeters
        "gender": ["M", "F", "M"],
        "activity_level": ["high", "medium", "low"],
        "health_goals": ["weight_loss", "muscle_gain", "endurance"]
    })

    # Load and preprocess fitness data using FitnessDatasetProcessor
    fitness_data = FitnessDatasetProcessor.load_fitness_data()
    fitness_data = FitnessDatasetProcessor.preprocess_fitness_data(fitness_data)

    # Clean nutrition data
    nutrition_data = nutrition_data.drop_duplicates(subset=['food']).reset_index(drop=True)
    for col in ['calories', 'fat', 'protein', 'carbohydrates']:
        if col in nutrition_data.columns:
            nutrition_data[col] = pd.to_numeric(nutrition_data[col], errors='coerce').fillna(0)

    fitness_data = fitness_data.drop_duplicates(subset=['name']).reset_index(drop=True)

    # Filter nutrition_data BEFORE generating interactions
    nutrition_data = nutrition_data[
        (nutrition_data['protein'] < 100) &
        (nutrition_data['calories'] < 1500) &
        (nutrition_data['fat'] < 100)
    ].reset_index(drop=True)

    # Initialize the fitness recommender
    fitness_recommender = FitnessRecommender()

    # Generate interactions for training
    fitness_interactions = np.random.randint(2, size=(len(user_data), len(fitness_data)))

    # Train the fitness recommender
    fitness_recommender.train(user_data, fitness_data, fitness_interactions)

    # Generate recommendations
    for i, user in user_data.iterrows():
        print(f"\n--- Fitness Recommendations for User {i+1} ---")
        recommendations = fitness_recommender.generate_recommendations(user)
        for _, rec in recommendations.iterrows():
            print(f"Activity: {rec['name']}, Level: {rec['level']}, Body Part: {rec['bodypart']}, Equipment: {rec['equipment']}")

if __name__ == "__main__":
    main()