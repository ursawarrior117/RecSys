import pandas as pd

class NutritionDatasetProcessor:
    @staticmethod
    def load_nutrition_data(file_path):
        """Load nutrition data from a CSV file."""
        return pd.read_csv(file_path)

    @staticmethod
    def preprocess_nutrition_data(data):
        """Preprocess nutrition data (e.g., handle missing values, normalize)."""
        data = data.dropna()  # Drop rows with missing values
        data['calories'] = data['calories'].astype(float)
        return data