import pandas as pd

def get_user_data():
    """
    Prompt the user to input their data and return as a DataFrame.
    """
    users = []
    while True:
        print("\nEnter user information:")
        age = int(input("Age: "))
        weight = float(input("Weight (kg): "))
        height = float(input("Height (cm): "))
        gender = input("Gender (M/F): ").strip().upper()
        activity_level = input("Activity level (low/medium/high): ").strip().lower()
        health_goals = input("Health goals(WL/MG): ").strip()
        sleep_good = int(input("Is your sleep good? (1 for Yes, 0 for No): "))
        users.append({
            "age": age,
            "weight": weight,
            "height": height,
            "gender": gender,
            "activity_level": activity_level,
            "health_goals": health_goals,
            "sleep_good": sleep_good
        })
        more = input("Add another user? (y/n): ").strip().lower()
        if more != 'y':
            break

    user_data = pd.DataFrame(users)
    user_data["user_id"] = range(1, len(user_data) + 1)
    return user_data