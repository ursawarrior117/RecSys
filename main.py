import numpy as np
import pandas as pd
from dataset_processor.fitness_processor import FitnessDatasetProcessor
from dataset_processor.nutrition_processor import NutritionDatasetProcessor
from NutritionModule import NutritionRecommender
from FitnessModule import FitnessRecommender

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

    # Load and preprocess datasets
    nutrition_data = NutritionDatasetProcessor.load_nutrition_data()
    nutrition_data = NutritionDatasetProcessor.preprocess_nutrition_data(nutrition_data)

    fitness_data = FitnessDatasetProcessor.load_fitness_data()
    fitness_data = FitnessDatasetProcessor.preprocess_fitness_data(fitness_data)
    
    # Initialize recommenders
    nutrition_recommender = NutritionRecommender()
    fitness_recommender = FitnessRecommender()

    # Generate interactions for training
    nutrition_interactions = np.random.randint(2, size=(len(user_data), len(nutrition_data)))
    fitness_interactions = np.random.randint(2, size=(len(user_data), len(fitness_data)))

    # Train recommenders
    nutrition_recommender.train(user_data, nutrition_data, nutrition_interactions)
    fitness_recommender.train(user_data, fitness_data, fitness_interactions)

    # Generate recommendations
    for i, user in user_data.iterrows():
        print(f"\n--- Recommendations for User {i+1} ---")
        
        # Generate fitness recommendations
        fitness_recommendations = fitness_recommender.generate_recommendations(user)
        print("Fitness Recommendations:")
        for _, rec in fitness_recommendations.iterrows():
            print(f"Activity: {rec['name']}, Level: {rec['level']}, Body Part: {rec['bodypart']}, Equipment: {rec['equipment']}")

        # Generate nutrition recommendations
        nutrition_recommendations = nutrition_recommender.generate_recommendations(user)
        print("Nutrition Recommendations:")
        for _, rec in nutrition_recommendations.iterrows():  # Ensure iteration over DataFrame rows
            print(f"Food: {rec['food']}, Calories: {rec['calories']}, Protein: {rec['protein']}g, Fat: {rec['fat']}g, Carbs: {rec['carbohydrates']}g")

        # Unified recommendations (example logic)
        print("\nUnified Recommendations:")
        for fitness_rec, (_, nutrition_rec) in zip(fitness_recommendations.head(3).itertuples(), nutrition_recommendations.iterrows()):
            print(f"Activity: {fitness_rec.name} (Level: {fitness_rec.level})")
            print(f"  Suggested Nutrition: {nutrition_rec['food']} - {nutrition_rec['calories']} calories, {nutrition_rec['protein']}g protein")
            print()

if __name__ == "__main__":
    main()