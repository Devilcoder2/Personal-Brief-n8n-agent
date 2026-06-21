# pyrefly: ignore [missing-import]
from sqlalchemy import create_engine, text
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base
Base = declarative_base()

# DB session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database Initializer
def init_db():
    # 1. Enable pgvector extension
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Drop unique constraint on briefings date if it exists
        try:
            conn.execute(text("ALTER TABLE briefings DROP CONSTRAINT IF EXISTS briefings_date_key;"))
        except Exception as e:
            print(f"Migration: Could not drop constraint briefings_date_key: {str(e)}")
        try:
            conn.execute(text("ALTER TABLE briefings DROP CONSTRAINT IF EXISTS uq_briefings_date;"))
        except Exception as e:
            print(f"Migration: Could not drop constraint uq_briefings_date: {str(e)}")
        try:
            conn.execute(text("DROP INDEX IF EXISTS ix_briefings_date;"))
        except Exception as e:
            print(f"Migration: Could not drop index ix_briefings_date: {str(e)}")
        
        # Add source_type column to briefings if it doesn't exist
        try:
            conn.execute(text("ALTER TABLE briefings ADD COLUMN IF NOT EXISTS source_type VARCHAR;"))
        except Exception as e:
            print(f"Migration: Could not add column source_type: {str(e)}")
            
        # Add unique constraint on (date, source_type) if it doesn't exist
        try:
            conn.execute(text("ALTER TABLE briefings ADD CONSTRAINT uq_briefings_date_source UNIQUE (date, source_type);"))
        except Exception as e:
            print(f"Migration: Could not add constraint uq_briefings_date_source: {str(e)}")
    
    # 2. Create tables
    Base.metadata.create_all(bind=engine)

