"""Script to reset database to original state (9 users, 73 interactions)."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from recsys_app.database.session import SessionLocal, init_sample_data
from recsys_app.database.models import User, Interaction

def reset_to_original():
    """Remove added users and interactions, keep only original 9 users and their interactions."""
    db = SessionLocal()
    try:
        # Original users are the first 9 (assuming added in order)
        all_users = db.query(User).order_by(User.id).all()
        if len(all_users) > 9:
            # Keep first 9, delete the rest
            users_to_delete = all_users[9:]
            for user in users_to_delete:
                # Delete interactions for these users
                db.query(Interaction).filter(Interaction.user_id == user.id).delete()
                db.delete(user)
            db.commit()
            print(f"Deleted {len(users_to_delete)} extra users and their interactions.")

        # Now, ensure interactions are only the original 73
        total_interactions = db.query(Interaction).count()
        if total_interactions > 73:
            # This shouldn't happen if we deleted user interactions, but just in case
            extra = total_interactions - 73
            # Delete most recent extra interactions
            interactions_to_delete = db.query(Interaction).order_by(Interaction.id.desc()).limit(extra).all()
            for inter in interactions_to_delete:
                db.delete(inter)
            db.commit()
            print(f"Deleted {extra} extra interactions.")

        print(f"Reset complete. Users: {db.query(User).count()}, Interactions: {db.query(Interaction).count()}")

    finally:
        db.close()

if __name__ == '__main__':
    reset_to_original()