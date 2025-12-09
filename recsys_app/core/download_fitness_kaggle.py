import os
import pandas as pd

def ensure_fitness_csv(local_path="data/fitness_data_kaggle.csv"):
    if os.path.exists(local_path):
        return local_path
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('niharika41298/gym-exercise-data', path='data', unzip=True)
        # Find the main CSV file
        for fname in os.listdir('data'):
            if fname.endswith('.csv') and 'exercise' in fname:
                os.rename(os.path.join('data', fname), local_path)
                return local_path
    except Exception as e:
        print(f"[Kaggle Download] Failed: {e}")
    return None
