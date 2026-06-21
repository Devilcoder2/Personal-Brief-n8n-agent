# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Optional, List, Dict, Any

class UserInterestBase(BaseModel):          
    interest_name: str
    tier: int
    weight: float

class UserInterestCreate(UserInterestBase):
    pass

class UserInterestResponse(UserInterestBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ArticleBase(BaseModel):
    title: str
    url: str
    author: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    source_type: str
    published_at: datetime
    article_metadata: Dict[str, Any] = Field(default_factory=dict)

class ArticleCreate(ArticleBase):
    pass

class ArticleResponse(ArticleBase):
    id: str
    created_at: datetime
    is_duplicate: bool
    duplicate_of_id: Optional[str] = None
    importance_score: Optional[float] = None
    relevance_score: Optional[float] = None
    novelty_score: Optional[float] = None
    overall_score: Optional[float] = None

    class Config:
        from_attributes = True

class BriefingCreate(BaseModel):
    date: date
    content: str

class BriefingResponse(BaseModel):
    id: str
    date: date
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

class IngestResponse(BaseModel):
    status: str
    scraped_count: int
    inserted_count: int
    skipped_count: int

class ProcessResponse(BaseModel):
    status: str
    processed_count: int
    duplicates_found: int
    ranked_count: int
