"""Script for inspecting database contents."""
import os
import sys
from pathlib import Path
from tabulate import tabulate

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from recsys_app.database.session import SessionLocal
from recsys_app.database.models import User, NutritionItem, FitnessItem

def inspect_db():
    """Display contents of database tables."""
    db = SessionLocal()
    
    try:
        # Inspect users
        users = db.query(User).all()
        print("\n=== Users ===")
        user_data = [[u.id, u.age, u.weight, u.height, u.gender, u.activity_level, u.health_goals]
                    for u in users]
        print(tabulate(user_data, headers=['ID', 'Age', 'Weight', 'Height', 'Gender', 'Activity', 'Goals']))
        
        # Inspect nutrition items
        nutrition = db.query(NutritionItem).all()
        print("\n=== Nutrition Items ===")
        nutrition_data = [[n.id, n.food, n.calories, n.protein, n.fat, n.carbohydrates]
                         for n in nutrition]
        print(tabulate(nutrition_data, headers=['ID', 'Food', 'Calories', 'Protein', 'Fat', 'Carbs']))
        
        # Inspect fitness items
        fitness = db.query(FitnessItem).all()
        print("\n=== Fitness Items ===")
        fitness_data = [[f.id, f.name, f.level, f.bodypart, f.equipment, f.type]
                       for f in fitness]
        print(tabulate(fitness_data, headers=['ID', 'Name', 'Level', 'Body Part', 'Equipment', 'Type']))
        
    finally:
        db.close()

if __name__ == "__main__":
    inspect_db()