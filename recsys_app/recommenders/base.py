"""Base recommender class for common functionality."""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

class BaseRecommender(ABC):
    def __init__(self):
        self.model = None
        self.feature_preprocessor = StandardScaler()
        self.database = None
        
    @abstractmethod
    def build_model(self, user_dim: int, item_dim: int):
        """Build the neural network model."""
        pass
        
    @abstractmethod
    def preprocess_data(self, user_data: pd.DataFrame, item_data: pd.DataFrame):
        """Preprocess user and item data."""
        pass
        
    @abstractmethod
    def train(self, user_data: pd.DataFrame, item_data: pd.DataFrame, interactions: np.ndarray):
        """Train the model."""
        pass
        
    @abstractmethod
    def predict_scores(self, user: dict, items: pd.DataFrame) -> np.ndarray:
        """Predict scores for a user and items."""
        pass
        
    @abstractmethod
    def generate_recommendations(self, user: dict, top_k: int = 5) -> pd.DataFrame:
        """Generate recommendations for a user."""
        pass