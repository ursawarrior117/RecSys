import os
import kaggle
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

    # Retain only the required columns
    required_columns = ["id", "activity_id", "name", "type", "level", "equipment","description","bodypart"]
    data = data[[col for col in required_columns if col in data.columns]]

    # Check if the DataFrame is empty
    if data.empty:
        print(f"Skipping file '{file_path}' as it does not contain the required columns.")
        return

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

# Fitness dataset configuration
fitness_dataset = {
    "kaggle_id": "niharika41298/gym-exercise-data",
    "table_name": "fitness_data"
}

# Connect to PostgreSQL
connection = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="super123",
    host="localhost",
    port="5432"
)

# Download and process the fitness dataset
dataset_path = download_dataset(fitness_dataset["kaggle_id"])
process_all_csv_files_in_folder(dataset_path, fitness_dataset["table_name"], connection)

# Close the connection
connection.close()

class FitnessDatasetProcessor:
    @staticmethod
    def load_fitness_data():
        """
        Load fitness data from the PostgreSQL database.
        """
        import psycopg2
        import pandas as pd

        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="super123",
            host="localhost",
            port="5432"
        )

        # Query to fetch all data from the fitness_data table
        query = "SELECT * FROM fitness_data"

        # Load the data into a pandas DataFrame
        fitness_data = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()

        return fitness_data
    @staticmethod
    def preprocess_fitness_data(data):
        """
        Preprocess the fitness data (e.g., normalize, clean).
        """
        data = data.fillna({"name": "", "type": "", "level": "", "equipment": "", "description": "", "bodypart": ""})
        data = data.dropna(subset=["name", "type", "level", "bodypart"])
        numeric_columns = ["id", "activity_id"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.fillna(0)
        return data