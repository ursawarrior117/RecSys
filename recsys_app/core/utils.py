"""Utility helpers for the RecSys application."""
from typing import Literal

def calculate_tdee(weight: float, height: float, age: int, gender: str, activity_level: str) -> float:
    """Estimate TDEE using Mifflin-St Jeor equation and activity multiplier.

    weight: kg, height: cm, age: years
    gender: 'M' or 'F'
    activity_level: 'low', 'medium', 'high'
    """
    try:
        w = float(weight)
        h = float(height)
        a = float(age)
    except Exception:
        return 2000.0

    if (gender or '').strip().upper() == 'M':
        bmr = 10*w + 6.25*h - 5*a + 5
    else:
        bmr = 10*w + 6.25*h - 5*a - 161

    mult = 1.2
    lvl = (activity_level or '').lower()
    if lvl == 'low':
        mult = 1.2
    elif lvl == 'medium':
        mult = 1.55
    elif lvl == 'high':
        mult = 1.725

    return float(bmr * mult)
