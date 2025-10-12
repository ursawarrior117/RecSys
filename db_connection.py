from sqlalchemy import create_engine, text

# Updated credentials
DB_USER = "postgres"
DB_PASSWORD = "super123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"

def get_engine():
    connection_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_url)
    return engine

def init_db():
    """Initialize the database by creating the interactions table if it doesn't exist."""
    engine = get_engine()
    with engine.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS interactions (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                item_id INTEGER,
                interaction INTEGER,
                timestamp TIMESTAMP
            )
        """))
        connection.commit()
