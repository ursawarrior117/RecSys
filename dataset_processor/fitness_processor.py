import pandas as pd

class FitnessDatasetProcessor:
    @staticmethod
    def load_fitness_data(file_path):
        """Load fitness data from a CSV file."""
        return pd.read_csv(file_path)

    @staticmethod
    def preprocess_fitness_data(data):
        """Preprocess fitness data (e.g., handle missing values, normalize)."""
        data = data.dropna()  # Drop rows with missing values
        data['calories_burned'] = data['calories_burned'].astype(float)
        return data