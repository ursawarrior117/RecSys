"""User-related API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database.session import get_db
from ..database.models import User
from ..schemas.schemas import UserCreate, UserResponse
from sqlalchemy import or_, text
import time
import random

# Configure router to not append trailing slash automatically
router = APIRouter()

@router.post("", response_model=UserResponse)
@router.post("/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user with optional custom ID."""
    # Ensure DB has the expected columns (upgrade if necessary)
    def ensure_user_columns(session: Session):
        # check existing columns
        res = session.execute(text("PRAGMA table_info('users')")).fetchall()
        cols = {r[1] for r in res}
        # Add missing columns via ALTER TABLE if needed
        if 'name' not in cols:
            try:
                session.execute(text("ALTER TABLE users ADD COLUMN name VARCHAR"))
                session.commit()
            except Exception:
                session.rollback()
        if 'external_id' not in cols:
            try:
                session.execute(text("ALTER TABLE users ADD COLUMN external_id VARCHAR"))
                session.commit()
            except Exception:
                session.rollback()

    ensure_user_columns(db)

    user_dict = user.dict()
    custom_id = user_dict.pop('id', None)

    name = user_dict.get('name')
    external_id = user_dict.get('external_id')
    
    # Auto-generate name if missing (allows old UI to create users)
    if not name or not name.strip():
        name = f"user_{int(time.time())}_{random.randint(100,999)}"
        user_dict['name'] = name
    
    # Auto-generate external_id if not provided
    if not external_id or not external_id.strip():
        external_id = f"ext_{int(time.time())}_{random.randint(100,999)}"
        user_dict['external_id'] = external_id
    
    user_dict['name'] = name.strip()

    # Check uniqueness: name must be unique, external_id must be unique
    existing_name = db.query(User).filter(User.name == user_dict['name']).first()
    if existing_name:
        raise HTTPException(status_code=400, detail=f"Name '{user_dict['name']}' already exists. Please use a unique name.")
    
    existing_external = db.query(User).filter(User.external_id == external_id).first()
    if existing_external:
        raise HTTPException(status_code=400, detail="External ID already exists")

    db_user = User(**user_dict)
    if custom_id is not None:
        db_user.id = custom_id

    db.add(db_user)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    db.refresh(db_user)
    return db_user


@router.delete("/clear")
def clear_all_users(db: Session = Depends(get_db)):
    """Delete ALL users and their interactions. Use with caution."""
    from ..database.models import Interaction

    try:
        # remove interactions first
        deleted_interactions = db.query(Interaction).delete()
        # then remove users
        deleted_users = db.query(User).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "cleared", "deleted_users": deleted_users, "deleted_interactions": deleted_interactions}


@router.delete("/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user and their interactions to free up space and reduce training load."""
    from ..database.models import Interaction

    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Remove interactions for the user first
    try:
        db.query(Interaction).filter(Interaction.user_id == user_id).delete()
    except Exception:
        db.rollback()

    # Delete the user
    try:
        db.delete(db_user)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "deleted", "id": user_id}

@router.get("", response_model=List[UserResponse])
@router.get("/", response_model=List[UserResponse])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all users."""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.get("/stats")
def get_users_with_stats(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get users with interaction counts (for pagination and activity display)."""
    from sqlalchemy import func
    from ..database.models import Interaction
    
    # query users with a count of their interactions
    users = db.query(User).offset(skip).limit(limit).all()
    result = []
    for user in users:
        interaction_count = db.query(func.count(Interaction.id)).filter(
            Interaction.user_id == user.id
        ).scalar() or 0
        user_dict = {k: v for k, v in user.__dict__.items() if not k.startswith('_')}
        user_dict['interaction_count'] = interaction_count
        result.append(user_dict)
    
    return result

@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get a specific user by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user: UserCreate, db: Session = Depends(get_db)):
    """Update a user's information."""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    for field, value in user.dict().items():
        setattr(db_user, field, value)

    db.commit()
    db.refresh(db_user)
    return db_user