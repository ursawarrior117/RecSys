import numpy as np
import pandas as pd
import csv
from dataset_processor.fitness_processor import FitnessDatasetProcessor
from dataset_processor.nutrition_processor import NutritionDatasetProcessor
from NutritionModule import NutritionRecommender, hybrid_nutrition_recommendations
from FitnessModule import FitnessRecommender
from user_input import get_user_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from db_connection import get_engine
from interaction_logger import log_user_interaction, load_interactions  # Using DB logging functions

engine = get_engine()

# -------------------------------------------------------------------
# 2. Build the interaction matrix from the database
def build_interaction_matrix(user_data, item_data):
    # Load interactions from the database using our helper function
    logs = load_interactions()
    if logs.empty:
        logs = pd.DataFrame(columns=["user_id", "item_id", "interaction"])
    # Keep only logs for available users and items
    logs = logs[logs["user_id"].isin(user_data["user_id"])]
    logs = logs[logs["item_id"].isin(item_data.index)]
    # Create matrix with user_id as index and item IDs as columns
    mat = pd.DataFrame(0, index=user_data["user_id"], columns=item_data.index)
    for _, row in logs.iterrows():
        mat.at[row["user_id"], row["item_id"]] = row["interaction"]
    return mat

# -------------------------------------------------------------------
# 3. Calculate user TDEE from basic features
def calculate_tdee(weight, height, age, gender, activity_level):
    if gender.upper() == "M":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    activity_factors = {"low": 1.2, "medium": 1.55, "high": 1.725}
    return bmr * activity_factors.get(activity_level.lower(), 1.2)

# -------------------------------------------------------------------
# Optional: simulation function for a warm-start (use only if no logs exist)
def simulate_interactions(user_data, item_data, item_type="nutrition"):
    interactions = np.zeros((len(user_data), len(item_data)))
    for i, user in user_data.iterrows():
        for j, item in item_data.iterrows():
            score = 0
            if item_type == "nutrition":
                if item["calories"] <= user["tdee"]:
                    score += 1
                if item["protein"] >= user["weight"] * (1.6 if user["health_goals"].upper() == "MG" else 1.2):
                    score += 1
                if item["fat"] < 30:
                    score += 1
                interactions[i, j] = 1 if score >= 2 else 0
            elif item_type == "fitness":
                if item["level"].lower() == user["activity_level"].lower():
                    score += 1
                if item["bodypart"].lower() in ["quadriceps", "hamstrings", "glutes"]:
                    score += 1
                interactions[i, j] = 1 if score >= 1 else 0
    return interactions

# -------------------------------------------------------------------
# 4. Evaluate the model on real interaction logs from the database
def evaluate_model(user_data, item_data, recommender):
    logs = load_interactions()
    if logs.empty:
        print("No logged interactions to evaluate.")
        return
    # Use an 80/20 split of the logs
    train_logs, test_logs = train_test_split(logs, test_size=0.2, random_state=42)
    
    # Helper: Build matrix from a provided logs DataFrame
    def build_matrix_from_logs(logs_df):
        m = pd.DataFrame(0, index=user_data["user_id"], columns=item_data.index)
        for _, row in logs_df.iterrows():
            m.at[row["user_id"], row["item_id"]] = row["interaction"]
        return m
    
    train_matrix = build_matrix_from_logs(train_logs)
    # Train recommender using interactions from DB
    recommender.train(user_data, item_data, train_matrix.values)
    
    y_true, y_pred = [], []
    for _, row in test_logs.iterrows():
        user = user_data[user_data["user_id"] == row["user_id"]].iloc[0]
        item_features = item_data.loc[[row["item_id"]]]
        score = recommender.predict_scores(user, item_features)
        y_true.append(row["interaction"])
        y_pred.append(1 if score[0] > 0.5 else 0)
    
    print("Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_pred))

# -------------------------------------------------------------------
def main():
    # Get user input; ensure get_user_data() returns a DataFrame with a "user_id" column.
    user_data = get_user_data()
    
    # Calculate TDEE and set default sleep_good if missing
    def process_user(row):
        row["tdee"] = calculate_tdee(row["weight"], row["height"], row["age"], row["gender"], row["activity_level"])
        if "sleep_good" not in row:
            row["sleep_good"] = 1
        return row
    user_data = user_data.apply(process_user, axis=1)
    
    for i, user in user_data.iterrows():
        print(f"User {user['user_id']} TDEE: {user['tdee']:.2f} kcal")
    
    # Load and preprocess datasets
    nutrition_data = NutritionDatasetProcessor.load_nutrition_data()
    nutrition_data = NutritionDatasetProcessor.preprocess_nutrition_data(nutrition_data)
    fitness_data = FitnessDatasetProcessor.load_fitness_data()
    fitness_data = FitnessDatasetProcessor.preprocess_fitness_data(fitness_data)
    
    # Clean and reset indexes
    nutrition_data = nutrition_data.drop_duplicates(subset=["food"]).reset_index(drop=True)
    for col in ["calories", "fat", "protein", "carbohydrates"]:
        if col in nutrition_data.columns:
            nutrition_data[col] = pd.to_numeric(nutrition_data[col], errors="coerce").fillna(0)
    fitness_data = fitness_data.drop_duplicates(subset=["name"]).reset_index(drop=True)
    
    # Filter nutrition data then reset index
    nutrition_data = nutrition_data[
        (nutrition_data["protein"] < 100) &
        (nutrition_data["calories"] < 1500) &
        (nutrition_data["fat"] < 100)
    ].reset_index(drop=True)
    
    # -------------------------------------------------------------------
    # Build interaction matrices from real logs stored in the database
    nutrition_interactions = build_interaction_matrix(user_data, nutrition_data)
    fitness_interactions = build_interaction_matrix(user_data, fitness_data)
    
    # If no real interactions exist, use simulation for bootstrapping
    if nutrition_interactions.values.sum() == 0:
        nutrition_interactions = simulate_interactions(user_data, nutrition_data, item_type="nutrition")
    if fitness_interactions.values.sum() == 0:
        fitness_interactions = simulate_interactions(user_data, fitness_data, item_type="fitness")
    
    # Initialize recommenders
    nutrition_recommender = NutritionRecommender()
    fitness_recommender = FitnessRecommender()
    
    # Ensure we have NumPy arrays (if already arrays, use them directly)
    train_nutrition = nutrition_interactions if isinstance(nutrition_interactions, np.ndarray) else nutrition_interactions.values
    train_fitness = fitness_interactions if isinstance(fitness_interactions, np.ndarray) else fitness_interactions.values
    
    # Train models with real or bootstrapped interactions
    nutrition_recommender.train(user_data, nutrition_data, train_nutrition)
    fitness_recommender.train(user_data, fitness_data, train_fitness)
    
    # Generate recommendations for each user
    for i, user in user_data.iterrows():
        print(f"\n--- Recommendations for User {user['user_id']} ---")
        # Fitness Recommendations
        fitness_recs = fitness_recommender.generate_recommendations(user, top_k=10)
        fitness_recs = fitness_recs.drop_duplicates(subset=["name"]).head(5)
        print("Fitness Recommendations:")
        for _, rec in fitness_recs.iterrows():
            print(f"Activity: {rec['name']}, Level: {rec['level']}, Body Part: {rec['bodypart']}, Equipment: {rec['equipment']}")
        
        # Nutrition Recommendations (hybrid and limiting based on TDEE and protein)
        nutrition_recs = hybrid_nutrition_recommendations(user, nutrition_data, nutrition_recommender, top_k=50, alpha=0.5)
        nutrition_recs = nutrition_recs.sort_values(["hybrid_score", "protein"], ascending=[False, False])
        if user["health_goals"].upper() == "MG":
            protein_limit = user["weight"] * 1.6
        else:
            protein_limit = user["weight"] * 1.2
        calorie_limit = user["tdee"]
        total_cals = 0
        total_protein = 0
        print("Nutrition Recommendations:")
        for _, rec in nutrition_recs.iterrows():
            if (total_cals + rec["calories"] > calorie_limit) or (total_protein + rec["protein"] > protein_limit):
                continue
            print(f"Food: {rec['food']}, Calories: {rec['calories']}, Protein: {rec['protein']}g, Fat: {rec['fat']}g, Carbs: {rec['carbohydrates']}g")
            total_cals += rec["calories"]
            total_protein += rec["protein"]
            if total_cals >= calorie_limit and total_protein >= protein_limit:
                break
        print(f"Total Calories Recommended: {total_cals:.1f}")
        print(f"Total Protein Recommended: {total_protein:.1f}g")
        if total_cals < calorie_limit:
            print(f"Warning: Total calories ({total_cals:.1f}) is below recommended ({calorie_limit:.1f}).")
        if total_protein < protein_limit:
            print(f"Warning: Total protein ({total_protein:.1f}g) is below recommended ({protein_limit:.1f}g).")
        
        # Example: When a user makes a selection, log that interaction.
        # log_user_interaction(user["user_id"], nutrition_recs.iloc[0].name, 1)
    
    # -------------------------------------------------------------------
    # Evaluate both models on real logged interactions from the database
    print("\n--- Evaluation ---")
    print("Nutrition Recommender Evaluation:")
    evaluate_model(user_data, nutrition_data, nutrition_recommender)
    print("Fitness Recommender Evaluation:")
    evaluate_model(user_data, fitness_data, fitness_recommender)

if __name__ == "__main__":
    main()