import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

class NutritionFitnessRecommender:
    def __init__(self, config=None):
        """
        Comprehensive recommender for nutrition and fitness recommendations
        """
        self.config = config or {}
        
        # Data preprocessors
        self.user_preprocessor = StandardScaler()
        self.nutrition_preprocessor = StandardScaler()
        self.activity_preprocessor = StandardScaler()
        
        # Multi-label encoders
        self.health_goal_encoder = MultiLabelBinarizer()
        self.fitness_level_encoder = MultiLabelBinarizer()
        
        # Will be set during preprocessing
        self.user_feature_dim = None
        self.food_feature_dim = None
        self.activity_feature_dim = None
        
        # Databases to store for recommendations
        self.nutrition_database = None
        self.activity_database = None
        
        # Hybrid recommendation models
        self.nutrition_model = None
        self.fitness_model = None
    
    def _build_nutrition_model(self):
        """
        Build neural network for nutrition recommendations
        """
        # Input layers
        user_input = keras.layers.Input(shape=(self.user_feature_dim,))
        food_input = keras.layers.Input(shape=(self.food_feature_dim,))
        
        # Collaborative and content-based filtering layers
        user_embedding = keras.layers.Dense(64, activation='relu')(user_input)
        food_embedding = keras.layers.Dense(64, activation='relu')(food_input)
        
        # Feature interaction
        interaction = keras.layers.Concatenate()([user_embedding, food_embedding])
        
        # Multi-layer processing
        x = keras.layers.Dense(128, activation='relu')(interaction)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        # Multiple outputs
        nutrition_recommendation = keras.layers.Dense(
            1, activation='sigmoid', name='nutrition_recommendation'
        )(x)
        
        nutritional_alignment = keras.layers.Dense(
            1, activation='linear', name='nutritional_alignment'
        )(x)
        
        model = keras.Model(
            inputs=[user_input, food_input],
            outputs=[nutrition_recommendation, nutritional_alignment]
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'nutrition_recommendation': 'binary_crossentropy',
                'nutritional_alignment': 'mean_squared_error'
            }
        )
        
        return model
    
    def _build_fitness_model(self):
        """
        Build neural network for fitness recommendations
        """
        # Input layers
        user_input = keras.layers.Input(shape=(self.user_feature_dim,))
        activity_input = keras.layers.Input(shape=(self.activity_feature_dim,))
        
        # Embedding layers
        user_embedding = keras.layers.Dense(64, activation='relu')(user_input)
        activity_embedding = keras.layers.Dense(64, activation='relu')(activity_input)
        
        # Feature interaction
        interaction = keras.layers.Concatenate()([user_embedding, activity_embedding])
        
        # Multi-layer processing
        x = keras.layers.Dense(128, activation='relu')(interaction)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        # Multiple outputs
        fitness_recommendation = keras.layers.Dense(
            1, activation='sigmoid', name='fitness_recommendation'
        )(x)
        
        fitness_intensity = keras.layers.Dense(
            1, activation='linear', name='fitness_intensity'
        )(x)
        
        calories_burned = keras.layers.Dense(
            1, activation='linear', name='calories_burned'
        )(x)
        
        model = keras.Model(
            inputs=[user_input, activity_input],
            outputs=[fitness_recommendation, fitness_intensity, calories_burned]
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'fitness_recommendation': 'binary_crossentropy',
                'fitness_intensity': 'mean_squared_error',
                'calories_burned': 'mean_squared_error'
            }
        )
        
        return model
    
    def preprocess_data(self, user_data, nutrition_data, activity_data):
        """
        Preprocess data for recommendation models
        """
        # Determine numerical feature columns dynamically
        numerical_features = ['age', 'weight', 'height']
        
        # Preprocess numerical features
        user_features = self.user_preprocessor.fit_transform(
            user_data[numerical_features]
        )
        
        # Encode health goals and fitness levels
        health_goal_encoded = self.health_goal_encoder.fit_transform(
            user_data['health_goals']
        )
        fitness_level_encoded = self.fitness_level_encoder.fit_transform(
            user_data['fitness_levels']
        )
        
        # Combine user features
        processed_user_features = np.hstack([
            user_features, 
            health_goal_encoded, 
            fitness_level_encoded
        ])
        
        # Update feature dimensions
        self.user_feature_dim = processed_user_features.shape[1]
        
        # Dynamically select nutrition and activity features
        nutrition_features = ['calories', 'protein', 'carbs', 'fat']
        activity_features = ['calories_burned', 'duration']
        
        # Preprocess nutrition and activity features
        processed_nutrition_features = self.nutrition_preprocessor.fit_transform(
            nutrition_data[nutrition_features]
        )
        self.food_feature_dim = processed_nutrition_features.shape[1]
        
        processed_activity_features = self.activity_preprocessor.fit_transform(
            activity_data[activity_features]
        )
        self.activity_feature_dim = processed_activity_features.shape[1]
        
        # Store databases for later recommendation generation
        self.nutrition_database = nutrition_data
        self.activity_database = activity_data
        
        # Build models after dimensions are known
        self.nutrition_model = self._build_nutrition_model()
        self.fitness_model = self._build_fitness_model()
        
        return processed_user_features, processed_nutrition_features, processed_activity_features
    
    def train(self, user_data, nutrition_data, activity_data, nutrition_interactions, fitness_interactions):
        """
        Train recommendation models
        """
        # Preprocess data
        processed_users, processed_nutrition, processed_activities = self.preprocess_data(
            user_data, nutrition_data, activity_data
        )
        
        # Prepare interaction pairs
        user_nutrition_pairs = np.repeat(processed_users, len(processed_nutrition), axis=0)
        nutrition_pairs = np.tile(processed_nutrition, (len(processed_users), 1))
        
        user_fitness_pairs = np.repeat(processed_users, len(processed_activities), axis=0)
        fitness_pairs = np.tile(processed_activities, (len(processed_users), 1))
        
        # Train nutrition model
        print("Training Nutrition Model...")
        self.nutrition_model.fit(
            [user_nutrition_pairs, nutrition_pairs],
            {
                'nutrition_recommendation': nutrition_interactions.flatten(),
                'nutritional_alignment': np.random.rand(len(user_nutrition_pairs), 1)
            },
            epochs=10,  # Reduced epochs for quicker testing
            batch_size=32,
            verbose=1
        )
        
        # Train fitness model
        print("Training Fitness Model...")
        self.fitness_model.fit(
            [user_fitness_pairs, fitness_pairs],
            {
                'fitness_recommendation': fitness_interactions.flatten(),
                'fitness_intensity': np.random.rand(len(user_fitness_pairs), 1),
                'calories_burned': np.random.rand(len(user_fitness_pairs), 1)
            },
            epochs=10,  # Reduced epochs for quicker testing
            batch_size=32,
            verbose=1
        )
    
    def generate_comprehensive_recommendations(self, user, top_k=3):
        """
        Generate integrated nutrition and fitness recommendations
        """
        # Preprocess user data
        processed_user, processed_nutrition, processed_activities = self.preprocess_data(
            pd.DataFrame([user]), 
            self.nutrition_database, 
            self.activity_database
        )
        
        # Generate nutrition recommendations
        nutrition_recommendations = self._generate_nutrition_recommendations(
            processed_user, processed_nutrition, top_k
        )
        
        # Generate fitness recommendations
        fitness_recommendations = self._generate_fitness_recommendations(
            processed_user, processed_activities, top_k
        )
        
        return {
            'nutrition_recommendations': nutrition_recommendations,
            'fitness_recommendations': fitness_recommendations
        }
    
    def _generate_nutrition_recommendations(self, processed_user, processed_nutrition, top_k):
        """
        Generate nutrition recommendations
        """
        user_nutrition_pairs = np.repeat(processed_user, len(processed_nutrition), axis=0)
        
        nutrition_recommendations, nutritional_scores = self.nutrition_model.predict([
            user_nutrition_pairs, 
            processed_nutrition
        ])
        
        # Sort and select top recommendations
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
    
    def _generate_fitness_recommendations(self, processed_user, processed_activities, top_k):
        """
        Generate fitness recommendations
        """
        user_fitness_pairs = np.repeat(processed_user, len(processed_activities), axis=0)
        
        fitness_recommendations, fitness_intensities, calories_burned = self.fitness_model.predict([
            user_fitness_pairs, 
            processed_activities
        ])
        
        # Sort and select top recommendations
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

def create_sample_dataset():
    """
    Create comprehensive sample dataset for nutrition and fitness
    """
    # User data
    user_data = pd.DataFrame({
        'age': [25, 35, 45],
        'weight': [70, 80, 90],
        'height': [170, 175, 180],
        'health_goals': [
            ['weight loss'], 
            ['muscle gain'], 
            ['general fitness']
        ],
        'fitness_levels': [
            ['beginner'], 
            ['intermediate'], 
            ['advanced']
        ]
    })
    
    # Nutrition database
    nutrition_data = pd.DataFrame({
        'food_id': range(1, 101),
        'name': [f'Food {i}' for i in range(1, 101)],
        'calories': np.random.randint(100, 500, 100),
        'protein': np.random.randint(5, 30, 100),
        'carbs': np.random.randint(10, 50, 100),
        'fat': np.random.randint(1, 20, 100)
    })
    
    # Activity database
    activity_data = pd.DataFrame({
        'activity_id': range(1, 51),
        'name': [f'Activity {i}' for i in range(1, 51)],
        'calories_burned': np.random.randint(50, 500, 50),
        'intensity': np.random.choice(['low', 'moderate', 'high'], 50),
        'duration': np.random.randint(15, 90, 50)
    })
    
    return user_data, nutrition_data, activity_data

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create sample dataset
    user_data, nutrition_data, activity_data = create_sample_dataset()
    
    # Initialize recommender
    recommender = NutritionFitnessRecommender()
    
    # Simulate interactions
    nutrition_interactions = np.random.randint(2, size=(len(user_data), len(nutrition_data)))
    fitness_interactions = np.random.randint(2, size=(len(user_data), len(activity_data)))
    
    # Train recommender
    recommender.train(
        user_data, 
        nutrition_data, 
        activity_data, 
        nutrition_interactions, 
        fitness_interactions
    )
    
    # Generate recommendations for each user
    for i, user in user_data.iterrows():
        print(f"\n--- Recommendations for User {i+1} ---")
        recommendations = recommender.generate_comprehensive_recommendations(user)
        
        print("Personalized Nutrition Recommendations:")
        for rec in recommendations['nutrition_recommendations']:
            print(f"Food: {rec['name']}")
            print(f"Recommendation Score: {rec['recommendation_score']:.2f}")
            print(f"Nutritional Alignment: {rec['nutritional_alignment']:.2f}")
            print("---")
        
        print("\nPersonalized Fitness Recommendations:")
        for rec in recommendations['fitness_recommendations']:
            print(f"Activity: {rec['name']}")
            print(f"Recommendation Score: {rec['recommendation_score']:.2f}")
            print(f"Intensity: {rec['intensity']:.2f}")
            print(f"Calories Burned: {rec['calories_burned']:.2f}")
            print("---")

if __name__ == "__main__":
    main()