import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataset_processor.fitness_processor import FitnessDatasetProcessor  # Import the processor

class FitnessRecommender:
    def __init__(self):
        self.user_preprocessor = StandardScaler()
        self.activity_preprocessor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Updated argument
        self.activity_database = None
        self.fitness_model = None

    def _build_fitness_model(self):
        """
        Build the fitness recommendation model.
        """
        user_input = keras.layers.Input(shape=(self.user_feature_dim,))
        activity_input = keras.layers.Input(shape=(self.activity_feature_dim,))
        user_embedding = keras.layers.Dense(64, activation='relu')(user_input)
        activity_embedding = keras.layers.Dense(64, activation='relu')(activity_input)
        interaction = keras.layers.Concatenate()([user_embedding, activity_embedding])
        x = keras.layers.Dense(128, activation='relu')(interaction)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        fitness_recommendation = keras.layers.Dense(1, activation='sigmoid', name='fitness_recommendation')(x)
        model = keras.Model(inputs=[user_input, activity_input], outputs=fitness_recommendation)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
        return model

    def preprocess_data(self, user_data, activity_data):
        """
        Preprocess user and activity data for training.
        """
        # Preprocess user data
        user_features = self.user_preprocessor.fit_transform(user_data[['age', 'weight', 'height']])

        # Preprocess activity data
        required_columns = ["level", "bodypart", "equipment", "type"]  # Metrics 1, 2, 3, 4
        activity_features = self.activity_preprocessor.fit_transform(activity_data[required_columns])

        # Store dimensions and activity database
        self.user_feature_dim = user_features.shape[1]
        self.activity_feature_dim = activity_features.shape[1]
        self.activity_database = activity_data

        # Build the fitness model
        self.fitness_model = self._build_fitness_model()

        return user_features, activity_features

    def train(self, user_data, activity_data, interactions):
        """
        Train the fitness recommendation model.
        """
        user_features, activity_features = self.preprocess_data(user_data, activity_data)

        # Create Cartesian product of users and activities
        user_features_repeated = np.repeat(user_features, len(activity_features), axis=0)
        activity_features_tiled = np.tile(activity_features, (len(user_features), 1))

        # Flatten interactions to match the Cartesian product
        interactions_flattened = interactions.flatten()

        # Train the model
        self.fitness_model.fit(
            [user_features_repeated, activity_features_tiled],
            interactions_flattened,
            epochs=10,
            batch_size=32
        )

    def generate_recommendations(self, user, top_k=5):
        """
        Generate fitness recommendations for a user.
        """
        # Convert the user data to a DataFrame with a single row
        user_features = self.user_preprocessor.transform(
            pd.DataFrame([user[['age', 'weight', 'height']].values], columns=['age', 'weight', 'height'])
        )

        # Transform activity features
        activity_features = self.activity_preprocessor.transform(self.activity_database[["level", "bodypart", "equipment", "type"]])

        # Repeat user features to match the number of activities
        user_features_repeated = np.repeat(user_features, len(activity_features), axis=0)

        # Predict scores for all activities
        scores = self.fitness_model.predict([user_features_repeated, activity_features], batch_size=128)

        # Get the indices of the top_k activities
        top_indices = scores.flatten().argsort()[-top_k:][::-1]

        # Return the top_k activities
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