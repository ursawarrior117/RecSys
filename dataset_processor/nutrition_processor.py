import pandas as pd
import psycopg2
from typing import List

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
    def preprocess_nutrition_data(nutrition_data: pd.DataFrame) -> pd.DataFrame:
        """
        Robust preprocessing for nutrition data:
         - ensure 'food' column exists
         - coerce numeric cols (calories, protein, fat, carbohydrates)
         - drop missing food rows
         - reset index and add stable 'item_id' column for logging/frontend
        """
        if nutrition_data is None or not isinstance(nutrition_data, pd.DataFrame):
            raise ValueError("nutrition_data must be a pandas DataFrame")

        if "food" not in nutrition_data.columns:
            raise ValueError("nutrition_data must contain a 'food' column")

        # Coerce numeric columns where present
        for col in ["calories", "protein", "fat", "carbohydrates"]:
            if col in nutrition_data.columns:
                nutrition_data[col] = pd.to_numeric(nutrition_data[col], errors="coerce").fillna(0.0)

        # Drop rows missing essential info
        nutrition_data = nutrition_data.dropna(subset=["food"]).copy()

        # Optional: require positive calories (if calories column exists)
        if "calories" in nutrition_data.columns:
            nutrition_data = nutrition_data[nutrition_data["calories"].astype(float) >= 0].copy()

        # Reset index and provide a stable item id used by frontend logging
        nutrition_data = nutrition_data.reset_index(drop=True)
        nutrition_data["item_id"] = nutrition_data.index.astype(int)

        return nutrition_data

    @staticmethod
    def ensure_columns(nutrition_data: pd.DataFrame, required: List[str]) -> pd.DataFrame:
        """Return a frame that contains at least the requested columns (adds missing as empty)."""
        df = nutrition_data.copy()
        for c in required:
            if c not in df.columns:
                df[c] = "" if df.shape[0] > 0 and df.dtypes.iloc[0] == object else 0
        return df