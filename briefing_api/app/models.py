import uuid
from datetime import datetime
# pyrefly: ignore [missing-import]
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON, Date, UniqueConstraint
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import relationship
# pyrefly: ignore [missing-import]
from pgvector.sqlalchemy import Vector
from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class UserInterest(Base):
    __tablename__ = "user_interests"

    id = Column(Integer, primary_key=True, index=True)
    interest_name = Column(String, unique=True, index=True, nullable=False)
    tier = Column(Integer, nullable=False) # 1, 2, or 3
    weight = Column(Float, nullable=False) # e.g., 1.0, 0.7, 0.4
    created_at = Column(DateTime, default=datetime.utcnow)

class Article(Base):
    __tablename__ = "articles"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, index=True, nullable=False)
    author = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    source_type = Column(String, index=True, nullable=False) # 'youtube', 'hn', 'reddit', 'github', 'arxiv', 'blog'
    published_at = Column(DateTime, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Deduplication
    is_duplicate = Column(Boolean, default=False, nullable=False)
    duplicate_of_id = Column(String, ForeignKey("articles.id"), nullable=True)
    
    # 768 dimensions for nomic-embed-text
    embedding = Column(Vector(768), nullable=True)

    # LLM Scoring
    importance_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    novelty_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)

    # Extra fields (views, stars, categories, raw data)
    article_metadata = Column(JSON, default=dict, nullable=False)

    # Self-referential relationship for duplicate articles
    duplicate_of = relationship("Article", remote_side=[id], backref="duplicates")

class Briefing(Base):
    __tablename__ = "briefings"

    id = Column(String, primary_key=True, default=generate_uuid)
    date = Column(Date, index=True, nullable=False)
    source_type = Column(String, index=True, nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('date', 'source_type', name='uq_briefings_date_source'),
    )
