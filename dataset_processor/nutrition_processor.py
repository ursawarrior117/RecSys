import pandas as pd
import psycopg2

# --- CONFIGURATION ---
CSV_PATH = r"d:\UrsaLa\RecSys\RecSys\data\2021-2023 FNDDS At A Glance - FNDDS Nutrient Values.csv"
TABLE_NAME = "nutrition_data"

COLUMN_MAPPING = {
    "Main food description": "food",
    "Energy (kcal)": "calories",
    "Total Fat (g)": "fat",
    "Fatty acids, total saturated (g)": "saturated_fats",
    "Fatty acids, total monounsaturated (g)": "monounsaturated_fats",
    "Fatty acids, total polyunsaturated (g)": "polyunsaturated_fats",
    "Carbohydrate (g)": "carbohydrates",
    "Sugars, total (g)": "sugars",
    "Protein (g)": "protein",
    "Fiber, total dietary (g)": "fiber",
    "Cholesterol (mg)": "cholesterol",
    "Sodium (mg)": "sodium",
    "Zinc (mg)": "zinc",
    "Calcium (mg)": "calcium",
    "Magnesium (mg)": "magnesium",
    "Caffeine (mg)": "caffeine"
}

NEEDED_COLUMNS = list(COLUMN_MAPPING.values())

# --- DATA LOADING & CLEANING ---
def load_and_clean_nutrition_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data.rename(columns=COLUMN_MAPPING)
    selected_cols = [col for col in NEEDED_COLUMNS if col in data.columns]
    data = data[selected_cols]
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    return data

# --- DATABASE INSERTION ---
def insert_into_postgresql(table_name, data, connection):
    cursor = connection.cursor()
    for _, row in data.iterrows():
        columns = ', '.join(row.index)
        placeholders = ', '.join(['%s'] * len(row))
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        try:
            cursor.execute(insert_query, tuple(row))
        except Exception as e:
            print(f"Error inserting row: {e}")
    connection.commit()
    cursor.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    nutrition_data = load_and_clean_nutrition_data(CSV_PATH)
    print("Cleaned nutrition data sample:")
    print(nutrition_data.head())

    # Connect to PostgreSQL and insert data
    connection = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="super123",
        host="localhost",
        port="5432"
    )
    insert_into_postgresql(TABLE_NAME, nutrition_data, connection)
    connection.close()
    print("Data inserted into PostgreSQL.")

class NutritionDatasetProcessor:
    @staticmethod
    def load_nutrition_data():
        # Connect to PostgreSQL and read nutrition_data table
        connection = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="super123",
            host="localhost",
            port="5432"
        )
        query = "SELECT * FROM nutrition_data"
        data = pd.read_sql_query(query, connection)
        connection.close()
        return data

    @staticmethod
    def preprocess_nutrition_data(data):
        # Any additional preprocessing steps
        return data