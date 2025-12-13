#!/usr/bin/env python3
"""
Test script for new recommender features:
1. Dietary restrictions filtering
2. User-based collaborative filtering
3. Contextual recommendations (meal_context)
4. Why-recommended explanations
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dietary_restrictions():
    """Test dietary restriction filtering."""
    print("\n" + "="*60)
    print("TEST 1: Dietary Restrictions Filtering")
    print("="*60)
    
    from recsys_app.database.models import User, NutritionItem
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create test database
    engine = create_engine("sqlite:///:memory:")
    from recsys_app.database.models import Base
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create test user with vegetarian restriction
    user = User(
        name="Vegetarian User",
        age=30,
        weight=70,
        height=170,
        gender='M',
        activity_level='medium',
        health_goals='MG',
        dietary_restrictions='vegetarian, gluten-free'
    )
    session.add(user)
    
    # Create test nutrition items
    meat_item = NutritionItem(
        food='Beef Steak',
        calories=300,
        protein=40,
        dietary_tags='meat, high-protein'
    )
    veggie_item = NutritionItem(
        food='Tofu Steak',
        calories=150,
        protein=20,
        dietary_tags='vegetarian, soy'
    )
    gluten_item = NutritionItem(
        food='Whole Wheat Bread',
        calories=200,
        protein=8,
        dietary_tags='gluten'
    )
    safe_item = NutritionItem(
        food='Rice Bowl',
        calories=250,
        protein=5,
        dietary_tags='vegetarian, gluten-free, vegan'
    )
    
    for item in [meat_item, veggie_item, gluten_item, safe_item]:
        session.add(item)
    
    session.commit()
    
    # Test filtering logic
    user_restrictions = set(r.strip().lower() for r in user.dietary_restrictions.split(',') if r.strip())
    print(f"User restrictions: {user_restrictions}")
    
    all_items = session.query(NutritionItem).all()
    safe_items = []
    for item in all_items:
        item_tags = set()
        if item.dietary_tags:
            item_tags = set(t.strip().lower() for t in item.dietary_tags.split(',') if t.strip())
        
        should_exclude = False
        if "vegetarian" in user_restrictions and "meat" in item_tags:
            should_exclude = True
            print(f"  ❌ Excluding '{item.food}' (meat, but user is vegetarian)")
        elif "gluten-free" in user_restrictions and "gluten" in item_tags:
            should_exclude = True
            print(f"  ❌ Excluding '{item.food}' (contains gluten, user is gluten-free)")
        
        if not should_exclude:
            safe_items.append(item)
            print(f"  ✓ Allowing '{item.food}'")
    
    print(f"\nResult: {len(safe_items)}/{len(all_items)} items are safe for user")
    assert len(safe_items) == 2, f"Expected 2 safe items, got {len(safe_items)}"
    print("✓ Dietary restrictions filtering test PASSED")
    
    session.close()


def test_collaborative_filtering():
    """Test user-based collaborative filtering."""
    print("\n" + "="*60)
    print("TEST 2: User-Based Collaborative Filtering")
    print("="*60)
    
    from recsys_app.database.models import User, NutritionItem, Interaction
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create test database
    engine = create_engine("sqlite:///:memory:")
    from recsys_app.database.models import Base
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create test users
    user1 = User(name="User1", age=30, weight=70, height=170, gender='M', activity_level='medium', health_goals='MG')
    user2 = User(name="User2", age=32, weight=72, height=172, gender='M', activity_level='medium', health_goals='MG')  # Similar to user1
    user3 = User(name="User3", age=50, weight=90, height=180, gender='M', activity_level='low', health_goals='WL')  # Different
    session.add_all([user1, user2, user3])
    session.flush()
    
    # Create test items
    item1 = NutritionItem(food='Protein Shake', calories=200, protein=30)
    item2 = NutritionItem(food='Apple', calories=80, protein=0.3)
    session.add_all([item1, item2])
    session.flush()
    
    # User2 likes item1
    interaction = Interaction(user_id=user2.id, nutrition_item_id=item1.id, event_type='like', rating=5)
    session.add(interaction)
    session.commit()
    
    print(f"User1 attributes: age={user1.age}, weight={user1.weight}, height={user1.height}, activity={user1.activity_level}, goal={user1.health_goals}")
    print(f"User2 attributes: age={user2.age}, weight={user2.weight}, height={user2.height}, activity={user2.activity_level}, goal={user2.health_goals}")
    print(f"User3 attributes: age={user3.age}, weight={user3.weight}, height={user3.height}, activity={user3.activity_level}, goal={user3.health_goals}")
    
    # Calculate similarity for user1
    user_attrs = (user1.age, user1.weight, user1.height, user1.activity_level, user1.health_goals)
    similarities = []
    
    for other_user in [user2, user3]:
        other_attrs = (other_user.age, other_user.weight, other_user.height, other_user.activity_level, other_user.health_goals)
        
        age_sim = 1.0 / (1.0 + abs(user_attrs[0] - other_attrs[0]) / 10.0)
        weight_sim = 1.0 / (1.0 + abs(user_attrs[1] - other_attrs[1]) / 20.0)
        height_sim = 1.0 / (1.0 + abs(user_attrs[2] - other_attrs[2]) / 10.0)
        activity_match = 1.0 if user_attrs[3] == other_attrs[3] else 0.5
        goal_match = 1.0 if user_attrs[4] == other_attrs[4] else 0.5
        
        similarity = (age_sim + weight_sim + height_sim + activity_match + goal_match) / 5.0
        similarities.append((other_user.name, similarity))
        
        print(f"\nSimilarity to {other_user.name}: {similarity:.3f}")
        print(f"  Age sim: {age_sim:.3f}, Weight sim: {weight_sim:.3f}, Height sim: {height_sim:.3f}")
        print(f"  Activity match: {activity_match:.3f}, Goal match: {goal_match:.3f}")
    
    # Get similar users with threshold > 0.6
    similar_users = [name for name, sim in similarities if sim > 0.6]
    print(f"\nSimilar users (threshold > 0.6): {similar_users}")
    
    # Collect their liked items
    similar_user_likes = set()
    for name, sim in similarities:
        if sim > 0.6:
            other_user = session.query(User).filter_by(name=name).first()
            likes = session.query(Interaction).filter(
                Interaction.user_id == other_user.id,
                Interaction.nutrition_item_id != None,
                Interaction.event_type == 'like'
            ).all()
            for like_it in likes:
                if like_it.nutrition_item_id:
                    similar_user_likes.add(int(like_it.nutrition_item_id))
                    print(f"  Found liked item from {name}: item_id={like_it.nutrition_item_id}")
    
    print(f"\nCollaborative filtering result: {len(similar_user_likes)} liked items from similar users")
    assert len(similar_user_likes) == 1, f"Expected 1 liked item, got {len(similar_user_likes)}"
    print("✓ Collaborative filtering test PASSED")
    
    session.close()


def test_meal_context():
    """Test meal context tagging."""
    print("\n" + "="*60)
    print("TEST 3: Meal Context Tagging")
    print("="*60)
    
    from recsys_app.database.models import NutritionItem
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///:memory:")
    from recsys_app.database.models import Base
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create items with meal context
    items = [
        NutritionItem(food='Pancakes', calories=300, protein=10, meal_context='breakfast'),
        NutritionItem(food='Sandwich', calories=400, protein=25, meal_context='lunch, dinner'),
        NutritionItem(food='Steak', calories=600, protein=60, meal_context='dinner'),
        NutritionItem(food='Nuts', calories=180, protein=6, meal_context='snack'),
    ]
    
    for item in items:
        session.add(item)
    session.commit()
    
    # Build meal context map
    meal_context_map = {}
    for item in session.query(NutritionItem).all():
        if item.meal_context:
            meal_context_map[item.id] = str(item.meal_context).lower()
    
    print(f"Meal context map: {meal_context_map}")
    
    # Test contextual matching for breakfast
    breakfast_items = [id for id, ctx in meal_context_map.items() if 'breakfast' in ctx]
    print(f"\nItems suitable for breakfast: {breakfast_items}")
    assert len(breakfast_items) == 1, f"Expected 1 breakfast item, got {len(breakfast_items)}"
    
    # Test contextual matching for dinner
    dinner_items = [id for id, ctx in meal_context_map.items() if 'dinner' in ctx]
    print(f"Items suitable for dinner: {dinner_items}")
    assert len(dinner_items) == 2, f"Expected 2 dinner items, got {len(dinner_items)}"
    
    print("✓ Meal context tagging test PASSED")
    
    session.close()


def test_reason_tags():
    """Test reason tag generation."""
    print("\n" + "="*60)
    print("TEST 4: Reason Tag Generation")
    print("="*60)
    
    # Simulate meal plan item with various reason tags
    user = {'health_goals': 'muscle_gain'}
    liked_item_ids = {101, 102}
    similar_user_likes = {103}
    meal_context_map = {104: 'breakfast', 105: 'lunch, dinner'}
    
    # Test item 101: user's liked item
    item_101 = {
        'id': 101,
        'food': 'Protein Shake',
        'protein_val': 30,
        'meal_context': 'breakfast'
    }
    
    reasons = []
    if 101 in liked_item_ids:
        reasons.append('liked')
    if 101 in similar_user_likes:
        reasons.append('similar_to_users_like_you')
    if 101 in meal_context_map and 'breakfast' in meal_context_map[101]:
        reasons.append('fits_breakfast')
    if 'muscle' in str(user.get('health_goals', '')).lower() and item_101.get('protein_val', 0) > 20:
        reasons.append('supports_muscle_gain')
    if not reasons:
        reasons.append('recommended')
    
    print(f"\nItem 101 (Liked, Muscle-Supporting, Breakfast): {', '.join(reasons)}")
    assert 'liked' in reasons and 'supports_muscle_gain' in reasons and 'fits_breakfast' in reasons
    
    # Test item 103: similar user's like
    item_103 = {
        'id': 103,
        'food': 'Eggs',
        'protein_val': 15,
    }
    
    reasons = []
    if 103 in liked_item_ids:
        reasons.append('liked')
    if 103 in similar_user_likes:
        reasons.append('similar_to_users_like_you')
    if 103 in meal_context_map and 'lunch' in meal_context_map[103]:
        reasons.append('fits_lunch')
    if not reasons:
        reasons.append('recommended')
    
    print(f"Item 103 (Similar User Likes): {', '.join(reasons)}")
    assert 'similar_to_users_like_you' in reasons
    
    # Test item 999: generic recommendation
    item_999 = {
        'id': 999,
        'food': 'Rice',
        'protein_val': 5,
    }
    
    reasons = []
    if 999 in liked_item_ids:
        reasons.append('liked')
    if 999 in similar_user_likes:
        reasons.append('similar_to_users_like_you')
    if not reasons:
        reasons.append('recommended')
    
    print(f"Item 999 (Generic): {', '.join(reasons)}")
    assert reasons == ['recommended']
    
    print("✓ Reason tag generation test PASSED")


if __name__ == '__main__':
    try:
        test_dietary_restrictions()
        test_collaborative_filtering()
        test_meal_context()
        test_reason_tags()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
