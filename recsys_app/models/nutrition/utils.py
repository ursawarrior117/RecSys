"""Utility functions for nutrition recommendations."""
import numpy as np
import pandas as pd

def hybrid_nutrition_recommendations(user: dict, nutrition_data: pd.DataFrame, recommender, top_k: int = 20, alpha: float = 0.5):
    """Generate hybrid nutrition recommendations combining collaborative and content-based filtering."""
    # Try collaborative predictions, but fall back to content-only if the recommender
    # or its preprocessors are not ready (e.g., not fitted or no model loaded).
    try:
        collab_scores = recommender.predict_scores(user, nutrition_data)
    except Exception:
        # Use uniform/neutral collaborative scores so content-based scoring dominates
        collab_scores = np.zeros(len(nutrition_data))
    
    # Content-based scoring based on nutritional metrics
    content_scores = (
        0.4 * nutrition_data['protein'] / (nutrition_data['calories'] + 1) +
        0.2 * nutrition_data['fiber'] / (nutrition_data['calories'] + 1) -
        0.2 * nutrition_data['fat'] / (nutrition_data['calories'] + 1) -
        0.2 * nutrition_data['sugars'] / (nutrition_data['calories'] + 1)
    )
    
    # Add magnesium boost for sleep issues
    if 'sleep_good' in user and user['sleep_good'] == 0 and 'magnesium' in nutrition_data.columns:
        content_scores += 0.2 * nutrition_data['magnesium'] / (nutrition_data['magnesium'].max() + 1e-8)
    
    # Normalize scores
    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
    
    # Combine scores
    hybrid_scores = alpha * collab_scores + (1 - alpha) * content_scores
    nutrition_data = nutrition_data.copy()
    nutrition_data['hybrid_score'] = hybrid_scores
    nutrition_data = nutrition_data.sort_values('hybrid_score', ascending=False)
    
    return nutrition_data.head(top_k)