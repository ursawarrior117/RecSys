import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

class NutritionRecommender:
    def __init__(self):
        self.user_feature_cols = ['age', 'weight', 'height', 'tdee', 'sleep_good']
        self.nutrition_feature_cols = [
            'calories', 'fat', 'protein', 'carbohydrates', 'fiber', 'sugars',
            'cholesterol', 'sodium', 'monounsaturated_fats', 'polyunsaturated_fats',
            'saturated_fats', 'zinc', 'calcium', 'magnesium', 'caffeine'
        ]
        self.user_preprocessor = StandardScaler()
        self.nutrition_preprocessor = StandardScaler()
        self.nutrition_database = None
        self.nutrition_model = None

    def build_model(self, user_dim, item_dim):
        input_user = keras.Input(shape=(user_dim,))
        input_item = keras.Input(shape=(item_dim,))
        x = keras.layers.Concatenate()([input_user, input_item])
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=[input_user, input_item], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.nutrition_model = model

    def preprocess_data(self, user_data, nutrition_data):
        nutrition_data = nutrition_data[
            (nutrition_data['protein'] < 100) &
            (nutrition_data['calories'] < 1500) &
            (nutrition_data['fat'] < 100)
        ]
        user_features = self.user_preprocessor.fit_transform(user_data[self.user_feature_cols])
        nutrition_features = self.nutrition_preprocessor.fit_transform(nutrition_data[self.nutrition_feature_cols])
        return user_features, nutrition_features

    def train(self, user_data, nutrition_data, interactions):
        user_features, nutrition_features = self.preprocess_data(user_data, nutrition_data)
        self.nutrition_database = nutrition_data.copy()
        if self.nutrition_model is None:
            self.build_model(user_features.shape[1], nutrition_features.shape[1])
        X_user = np.repeat(user_features, nutrition_features.shape[0], axis=0)
        X_item = np.tile(nutrition_features, (user_features.shape[0], 1))
        y = interactions.flatten()
        self.nutrition_model.fit([X_user, X_item], y, epochs=5, batch_size=32, verbose=0)

    def generate_recommendations(self, user, top_k=5):
        user_df = pd.DataFrame([user])[self.user_feature_cols]
        user_features = self.user_preprocessor.transform(user_df)
        nutrition_features = self.nutrition_preprocessor.transform(self.nutrition_database[self.nutrition_feature_cols])
        user_features_repeated = np.repeat(user_features, len(nutrition_features), axis=0)
        scores = self.nutrition_model.predict([user_features_repeated, nutrition_features], batch_size=128, verbose=0)
        top_indices = scores.flatten().argsort()[-top_k:][::-1]
        return self.nutrition_database.iloc[top_indices]

    def predict_scores(self, user, nutrition_data):
        user_df = pd.DataFrame([user])[self.user_feature_cols]
        user_features = self.user_preprocessor.transform(
            pd.concat([user_df]*len(nutrition_data), ignore_index=True)
        )
        nutrition_features = self.nutrition_preprocessor.transform(
            nutrition_data[self.nutrition_feature_cols]
        )
        scores = self.nutrition_model.predict([user_features, nutrition_features], verbose=0)
        return scores.flatten()

def hybrid_nutrition_recommendations(user, nutrition_data, recommender, top_k=20, alpha=0.5):
    collab_scores = recommender.predict_scores(user, nutrition_data)
    content_scores = (
        0.4 * nutrition_data['protein'] / (nutrition_data['calories'] + 1) +
        0.2 * nutrition_data['fiber'] / (nutrition_data['calories'] + 1) -
        0.2 * nutrition_data['fat'] / (nutrition_data['calories'] + 1) -
        0.2 * nutrition_data['sugars'] / (nutrition_data['calories'] + 1)
    )
    if 'sleep_good' in user and user['sleep_good'] == 0 and 'magnesium' in nutrition_data.columns:
        content_scores += 0.2 * nutrition_data['magnesium'] / (nutrition_data['magnesium'].max() + 1e-8)
    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
    hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores
    nutrition_data = nutrition_data.copy()
    nutrition_data['hybrid_score'] = hybrid_scores
    nutrition_data = nutrition_data.sort_values('hybrid_score', ascending=False)
    return nutrition_data