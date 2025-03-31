import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

class FitnessRecommender:
    def __init__(self):
        self.user_preprocessor = StandardScaler()
        self.activity_preprocessor = StandardScaler()
        self.fitness_level_encoder = MultiLabelBinarizer()
        self.user_feature_dim = None
        self.activity_feature_dim = None
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
        fitness_recommendation = keras.layers.Dense(1, activation='sigmoid', name='fitness_recommendation')(x)
        fitness_intensity = keras.layers.Dense(1, activation='linear', name='fitness_intensity')(x)
        calories_burned = keras.layers.Dense(1, activation='linear', name='calories_burned')(x)
        model = keras.Model(inputs=[user_input, activity_input], outputs=[fitness_recommendation, fitness_intensity, calories_burned])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'fitness_recommendation': 'binary_crossentropy',
                'fitness_intensity': 'mean_squared_error',
                'calories_burned': 'mean_squared_error'
            }
        )
        return model

    def preprocess_data(self, user_data, activity_data):
        user_features = self.user_preprocessor.fit_transform(user_data[['age', 'weight', 'height']])
        fitness_level_encoded = self.fitness_level_encoder.fit_transform(user_data['fitness_levels'])
        processed_user_features = np.hstack([user_features, fitness_level_encoded])
        self.user_feature_dim = processed_user_features.shape[1]
        processed_activity_features = self.activity_preprocessor.fit_transform(activity_data[['calories_burned', 'duration']])
        self.activity_feature_dim = processed_activity_features.shape[1]
        self.activity_database = activity_data
        self.fitness_model = self._build_fitness_model()
        return processed_user_features, processed_activity_features

    def train(self, user_data, activity_data, fitness_interactions):
        processed_users, processed_activities = self.preprocess_data(user_data, activity_data)
        user_fitness_pairs = np.repeat(processed_users, len(processed_activities), axis=0)
        fitness_pairs = np.tile(processed_activities, (len(processed_users), 1))
        self.fitness_model.fit(
            [user_fitness_pairs, fitness_pairs],
            {
                'fitness_recommendation': fitness_interactions.flatten(),
                'fitness_intensity': np.random.rand(len(user_fitness_pairs), 1),
                'calories_burned': np.random.rand(len(user_fitness_pairs), 1)
            },
            epochs=10,
            batch_size=32,
            verbose=1
        )

    def generate_recommendations(self, user, top_k=3):
        processed_user, processed_activities = self.preprocess_data(pd.DataFrame([user]), self.activity_database)
        user_fitness_pairs = np.repeat(processed_user, len(processed_activities), axis=0)
        fitness_recommendations, fitness_intensities, calories_burned = self.fitness_model.predict([user_fitness_pairs, processed_activities])
        sorted_indices = np.argsort(fitness_recommendations.flatten())[::-1][:top_k]
        return [
            {
                'activity_id': self.activity_database.iloc[i]['activity_id'],
                'name': self.activity_database.iloc[i]['name'],
                'recommendation_score': fitness_recommendations[i][0],
                'intensity': fitness_intensities[i][0],
                'calories_burned': calories_burned[i][0]
            }
            for i in sorted_indices
        ]