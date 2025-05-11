import os
import kagglehub
import pandas as pd
import psycopg2

def download_dataset(dataset_name):
    """
    Download a dataset from Kaggle using the Kaggle API.
    """
    print(f"Downloading dataset: {dataset_name}")
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded to: {path}")
    return path

def insert_into_postgresql(table_name, file_path, connection):
    """
    Insert data from a CSV file into a PostgreSQL table.
    """
    # Load CSV into a DataFrame
    data = pd.read_csv(file_path)

    # Drop unnecessary columns (e.g., unnamed columns)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Rename columns to valid PostgreSQL column names
    data.columns = [col.strip().replace(" ", "_").replace(":", "_").lower() for col in data.columns]

    # Map dataset-specific column names to match the database schema
    column_mapping = {
        "Caloric Value": "calories",
        "Fat": "fat",
        "Saturated Fats": "saturated_fats",
        "Monounsaturated Fats": "monounsaturated_fats",
        "Polyunsaturated Fats": "polyunsaturated_fats",
        "Carbohydrates": "carbohydrates",
        "Sugars": "sugars",
        "Protein": "protein",
        "Dietary Fiber": "fiber",
        "Cholesterol": "cholesterol",
        "Sodium": "sodium"
    }
    data.rename(columns=column_mapping, inplace=True)

    # Retain only the required columns
    required_columns = [
        "id",
        "food",
        "calories",
        "fat",
        "saturated_fats",
        "monounsaturated_fats",
        "polyunsaturated_fats",
        "carbohydrates",
        "sugars",
        "protein",
        "fiber",
        "cholesterol",
        "sodium"
    ]
    data = data[[col for col in required_columns if col in data.columns]]

    # Debug: Print the first few rows of the DataFrame
    print(f"Data from file '{file_path}':")
    print(data.head())

    print("DataFrame before numeric conversion:")
    print(data.head())
    print(data.dtypes)

    # Ensure numeric columns are properly converted
    numeric_columns = [
        "calories",
        "fat",
        "saturated_fats",
        "monounsaturated_fats",
        "polyunsaturated_fats",
        "carbohydrates",
        "sugars",
        "protein",
        "fiber",
        "cholesterol",
        "sodium"
    ]
    for col in numeric_columns:
        if col in data.columns:
            print(f"Converting column '{col}' to numeric.")  # Debug statement
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, set invalid values to NaN
        else:
            print(f"Column '{col}' not found in the DataFrame.")  # Debug statement

    # Ensure all required columns exist in the DataFrame
    for col in numeric_columns:
        if col not in data.columns:
            print(f"Column '{col}' is missing. Adding it with default value 0.")  # Debug statement
            data[col] = 0

    print("DataFrame after adding missing columns:")
    print(data.head())
    print(data.dtypes)

    # Handle missing values
    data = data.fillna(0)

    # Insert data into PostgreSQL
    cursor = connection.cursor()
    for _, row in data.iterrows():
        columns = ', '.join(row.index)
        values = ', '.join(['%s'] * len(row))
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
        cursor.execute(insert_query, tuple(row))
    connection.commit()
    print(f"Inserted {len(data)} records from '{file_path}' into the '{table_name}' table.")

def process_all_csv_files_in_folder(folder_path, table_name, connection):
    """
    Process all CSV files in a folder and insert them into a PostgreSQL table.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")
            insert_into_postgresql(table_name, file_path, connection)

# Nutrition dataset configuration
nutrition_dataset = {
    "kaggle_id": "utsavdey1410/food-nutrition-dataset",
    "table_name": "nutrition_data"
}

# Connect to PostgreSQL
connection = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="super123",
    host="localhost",
    port="5432"
)

# Download and process the nutrition dataset
dataset_path = download_dataset(nutrition_dataset["kaggle_id"])
process_all_csv_files_in_folder(dataset_path, nutrition_dataset["table_name"], connection)

# Close the connection
connection.close()

class NutritionDatasetProcessor:
    @staticmethod
    def load_nutrition_data():
        """
        Load nutrition data from the PostgreSQL database.
        """
        from sqlalchemy import create_engine
        import pandas as pd

        # Create a SQLAlchemy engine
        engine = create_engine("postgresql+psycopg2://postgres:super123@localhost:5432/postgres")

        # Query to fetch all data from the nutrition_data table
        query = "SELECT * FROM nutrition_data"

        # Load the data into a pandas DataFrame
        nutrition_data = pd.read_sql_query(query, engine)

        return nutrition_data

    @staticmethod
    def preprocess_nutrition_data(data):
        """
        Preprocess the nutrition data (e.g., normalize, clean).
        """
        # Rename columns to match expected names
        column_mapping = {
            "Caloric Value": "calories",
            "Fat": "fat",
            "Saturated Fats": "saturated_fats",
            "Monounsaturated Fats": "monounsaturated_fats",
            "Polyunsaturated Fats": "polyunsaturated_fats",
            "Carbohydrates": "carbohydrates",
            "Sugars": "sugars",
            "Protein": "protein",
            "Dietary Fiber": "fiber",
            "Cholesterol": "cholesterol",
            "Sodium": "sodium"
        }
        data.rename(columns=column_mapping, inplace=True)

        # Ensure numeric columns are properly converted
        numeric_columns = [
            "calories",
            "fat",
            "saturated_fats",
            "monounsaturated_fats",
            "polyunsaturated_fats",
            "carbohydrates",
            "sugars",
            "protein",
            "fiber",
            "cholesterol",
            "sodium"
        ]

        # Add missing columns with default values
        for col in numeric_columns:
            if col not in data.columns:
                print(f"Column '{col}' is missing. Adding it with default value 0.")  # Debug statement
                data[col] = pd.Series([0] * len(data), index=data.index)  # Add column as a Series

        # Convert columns to numeric
        for col in numeric_columns:
            if col in data.columns:
                print(f"Converting column '{col}' to numeric.")  # Debug statement
                data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, set invalid values to NaN
                if data[col].isna().any():
                    print(f"Warning: Column '{col}' contains non-numeric values that were coerced to NaN.")

        # Handle missing values
        data = data.fillna(0)

        # Debug: Print the DataFrame after preprocessing
        print("DataFrame after preprocessing:")
        print(data.head())
        print(data.dtypes)

        return data

    def preprocess_data(self, user_data, nutrition_data):
        """
        Preprocess user and nutrition data for training.
        """
        # Preprocess user data (only using 'age')
        user_features = self.user_preprocessor.fit_transform(user_data[['age']])

        # Preprocess nutrition data
        required_columns = [
            "calories",
            "fat",
            "saturated_fats",
            "monounsaturated_fats",
            "polyunsaturated_fats",
            "carbohydrates",
            "sugars",
            "protein",
            "fiber",
            "cholesterol",
            "sodium"
        ]
        processed_nutrition_features = self.nutrition_preprocessor.fit_transform(nutrition_data[required_columns])
        self.user_feature_dim = user_features.shape[1]
        self.food_feature_dim = processed_nutrition_features.shape[1]

        # Store the nutrition database for recommendations
        self.nutrition_database = nutrition_data
        self.nutrition_model = self._build_nutrition_model()

        return user_features, processed_nutrition_features