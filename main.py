import numpy as np
from dataset_processor.fitness_processor import FitnessDatasetProcessor
from dataset_processor.nutrition_processor import NutritionDatasetProcessor
from NutritionModule import NutritionRecommender
from FitnessModule import FitnessRecommender
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Load and preprocess datasets
    nutrition_data = NutritionDatasetProcessor.load_nutrition_data("nutrition_data.csv")
    nutrition_data = NutritionDatasetProcessor.preprocess_nutrition_data(nutrition_data)

    fitness_data = FitnessDatasetProcessor.load_fitness_data("fitness_data.csv")
    fitness_data = FitnessDatasetProcessor.preprocess_fitness_data(fitness_data)

    # Create sample user data
    user_data = pd.DataFrame({
        'age': [25, 35, 45],
        'weight': [70, 80, 90],
        'height': [170, 175, 180],
        'health_goals': [['weight loss'], ['muscle gain'], ['general fitness']],
        'fitness_levels': [['beginner'], ['intermediate'], ['advanced']]
    })

    # Initialize recommenders
    nutrition_recommender = NutritionRecommender()
    fitness_recommender = FitnessRecommender()

    # Train recommenders
    nutrition_interactions = np.random.randint(2, size=(len(user_data), len(nutrition_data)))
    fitness_interactions = np.random.randint(2, size=(len(user_data), len(fitness_data)))
    nutrition_recommender.train(user_data, nutrition_data, nutrition_interactions)
    fitness_recommender.train(user_data, fitness_data, fitness_interactions)

    # Generate recommendations
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