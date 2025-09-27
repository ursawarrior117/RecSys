import numpy as np
import pandas as pd
from dataset_processor.fitness_processor import FitnessDatasetProcessor
from dataset_processor.nutrition_processor import NutritionDatasetProcessor
from NutritionModule import NutritionRecommender, hybrid_nutrition_recommendations
from FitnessModule import FitnessRecommender
from user_input import get_user_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def calculate_tdee(weight, height, age, gender, activity_level):
    if gender.upper() == 'M':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    activity_factors = {
        'low': 1.2,
        'medium': 1.55,
        'high': 1.725
    }
    return bmr * activity_factors.get(activity_level.lower(), 1.2)

def simulate_interactions(user_data, item_data, item_type="nutrition"):
    interactions = np.zeros((len(user_data), len(item_data)))
    for i, user in user_data.iterrows():
        for j, item in item_data.iterrows():
            score = 0
            if item_type == "nutrition":
                # Improved simulation: calories, protein, fat
                if item['calories'] <= user['tdee']:
                    score += 1
                if item['protein'] >= user['weight'] * (1.6 if user['health_goals'].upper() == 'MG' else 1.2):
                    score += 1
                if item['fat'] < 30:
                    score += 1
                interactions[i, j] = 1 if score >= 2 else 0
            elif item_type == "fitness":
                if item['level'].lower() == user['activity_level'].lower():
                    score += 1
                if item['bodypart'].lower() in ['quadriceps', 'hamstrings', 'glutes']:
                    score += 1
                interactions[i, j] = 1 if score >= 1 else 0
    return interactions

def main():
    # Get user input
    user_data = get_user_data()

    # Calculate TDEE for each user
    def process_user_row(row):
        row['tdee'] = calculate_tdee(row['weight'], row['height'], row['age'], row['gender'], row['activity_level'])
        if 'sleep_good' not in row:
            row['sleep_good'] = 1
        return row

    user_data = user_data.apply(process_user_row, axis=1)

    for i, user in user_data.iterrows():
        print(f"User {i+1} TDEE: {user['tdee']:.2f} kcal")

    # Load and preprocess datasets
    nutrition_data = NutritionDatasetProcessor.load_nutrition_data()
    nutrition_data = NutritionDatasetProcessor.preprocess_nutrition_data(nutrition_data)
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

    # Generate interactions
    nutrition_interactions = simulate_interactions(user_data, nutrition_data, item_type="nutrition")
    fitness_interactions = simulate_interactions(user_data, fitness_data, item_type="fitness")

    # Initialize recommenders
    nutrition_recommender = NutritionRecommender()
    fitness_recommender = FitnessRecommender()

    # Train models
    nutrition_recommender.train(user_data, nutrition_data, nutrition_interactions)
    fitness_recommender.train(user_data, fitness_data, fitness_interactions)

    # Recommendations
    for i, user in user_data.iterrows():
        print(f"\n--- Recommendations for User {i+1} ---")
        # Fitness recommendations
        fitness_recommendations = fitness_recommender.generate_recommendations(user, top_k=10)
        fitness_recommendations = fitness_recommendations.drop_duplicates(subset=['name']).head(5)
        print("Fitness Recommendations:")
        for _, rec in fitness_recommendations.iterrows():
            print(f"Activity: {rec['name']}, Level: {rec['level']}, Body Part: {rec['bodypart']}, Equipment: {rec['equipment']}")

        # Nutrition recommendations (hybrid, then limit by TDEE and protein needs)
        nutrition_recommendations = hybrid_nutrition_recommendations(user, nutrition_data, nutrition_recommender, top_k=50, alpha=0.5)
        # Sort by hybrid score, then by protein descending
        nutrition_recommendations = nutrition_recommendations.sort_values(['hybrid_score', 'protein'], ascending=[False, False])

        # Set protein limit based on health goal
        if user['health_goals'].upper() == 'MG':
            protein_limit = user['weight'] * 1.6
        else:
            protein_limit = user['weight'] * 1.2

        calorie_limit = user['tdee']
        total_cals = 0
        total_protein = 0
        total_fat = 0
        total_carb = 0
        total_magnesium = 0


        print("Nutrition Recommendations:")
        for _, rec in nutrition_recommendations.iterrows():
            # If adding this food would exceed either limit, skip it
            if (total_cals + rec['calories'] > calorie_limit) or (total_protein + rec['protein'] > protein_limit):
                continue
            print(f"Food: {rec['food']}, Calories: {rec['calories']}, Protein: {rec['protein']}g, Fat: {rec['fat']}g, Carbs: {rec['carbohydrates']}g")
            total_cals += rec['calories']
            total_protein += rec['protein']
            total_fat += rec.get('fat', 0) or 0
            total_carb += rec.get('carbohydrates', 0) or 0
            total_magnesium += rec.get('magnesium', 0) or 0

            # Stop if both minimums are reached
            if total_cals >= calorie_limit and total_protein >= protein_limit:
                break
        print(f"Total Calories Recommended: {total_cals:.1f}")
        print(f"Total Protein Recommended: {total_protein:.1f}g")
        print(f"Total Fat Recommended: {total_fat:.1f}g")
        print(f"Total Carbohydrates Recommended: {total_carb:.1f}g")
        print(f"Total Magnesiuum Recommended: {total_magnesium:.1f}g")
        if total_cals < calorie_limit:
            print(f"Warning: Total calories ({total_cals:.1f}) is below recommended ({calorie_limit:.1f}) for your goal.")
        if total_protein < protein_limit:
            print(f"Warning: Total protein ({total_protein:.1f}g) is below recommended ({protein_limit:.1f}g) for your goal.")


if __name__ == "__main__":
    main()