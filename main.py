import numpy as np
import pandas as pd
from NutritionModule import NutritionRecommender
from FitnessModule import FitnessRecommender

def create_sample_dataset():
    user_data = pd.DataFrame({
        'age': [25, 35, 45],
        'weight': [70, 80, 90],
        'height': [170, 175, 180],
        'health_goals': [['weight loss'], ['muscle gain'], ['general fitness']],
        'fitness_levels': [['beginner'], ['intermediate'], ['advanced']]
    })
    nutrition_data = pd.DataFrame({
        'food_id': range(1, 101),
        'name': [f'Food {i}' for i in range(1, 101)],
        'calories': np.random.randint(100, 500, 100),
        'protein': np.random.randint(5, 30, 100),
        'carbs': np.random.randint(10, 50, 100),
        'fat': np.random.randint(1, 20, 100)
    })
    activity_data = pd.DataFrame({
        'activity_id': range(1, 51),
        'name': [f'Activity {i}' for i in range(1, 51)],
        'calories_burned': np.random.randint(50, 500, 50),
        'intensity': np.random.choice(['low', 'moderate', 'high'], 50),
        'duration': np.random.randint(15, 90, 50)
    })
    return user_data, nutrition_data, activity_data

def main():
    np.random.seed(42)
    user_data, nutrition_data, activity_data = create_sample_dataset()
    nutrition_recommender = NutritionRecommender()
    fitness_recommender = FitnessRecommender()
    nutrition_interactions = np.random.randint(2, size=(len(user_data), len(nutrition_data)))
    fitness_interactions = np.random.randint(2, size=(len(user_data), len(activity_data)))
    nutrition_recommender.train(user_data, nutrition_data, nutrition_interactions)
    fitness_recommender.train(user_data, activity_data, fitness_interactions)
    for i, user in user_data.iterrows():
        print(f"\n--- Recommendations for User {i+1} ---")
        print("Nutrition Recommendations:")
        for rec in nutrition_recommender.generate_recommendations(user):
            print(f"Food: {rec['name']}, Score: {rec['recommendation_score']:.2f}")
        print("Fitness Recommendations:")
        for rec in fitness_recommender.generate_recommendations(user):
            print(f"Activity: {rec['name']}, Score: {rec['recommendation_score']:.2f}")

if __name__ == "__main__":
    main()