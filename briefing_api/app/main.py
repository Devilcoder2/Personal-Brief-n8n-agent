# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Depends, HTTPException, Query
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from app.database import get_db, init_db
from app.models import Article, Briefing, UserInterest
from app.schemas import IngestResponse, ProcessResponse, BriefingResponse, ArticleResponse
from app.scrapers import SCRAPERS
from app.services import dedup_service, ranking_service, insights_service, embedding_service
from app.config import settings

app = FastAPI(
    title="Daily Tech Intelligence Briefing API",
    description="Helper service that manages scraping, semantic deduplication, LLM-based scoring, and briefing compilation.",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    """
    Automatically initialize database schema and extensions on start.
    """
    try:
        init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error during startup database initialization: {str(e)}")

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "time": datetime.utcnow(),
        "config": {
            "embed_model": settings.OLLAMA_EMBED_MODEL,
            "llm_model": settings.OLLAMA_LLM_MODEL,
            "ollama_url": settings.OLLAMA_BASE_URL
        }
    }

@app.post("/init-db")
def trigger_init_db():
    """
    Manually trigger database table creation and pgvector activation.
    """
    try:
        init_db()
        return {"status": "success", "message": "Database tables and pgvector extension initialized."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
def trigger_ingest(db: Session = Depends(get_db)):
    """
    Runs all platform scrapers and inserts newly found tech news into the database.
    """
    scraped_total = 0
    inserted_count = 0
    skipped_count = 0

    for scraper in SCRAPERS:
        try:
            print(f"Starting ingestion from scraper: {scraper.name}")
            articles = scraper.scrape()
            scraped_total += len(articles)
            
            for art_data in articles:
                # Check if article URL already exists to skip duplicates
                existing = db.query(Article).filter(Article.url == art_data.url).first()
                if existing:
                    skipped_count += 1
                    continue
                
                # Insert new article
                db_article = Article(
                    title=art_data.title,
                    url=art_data.url,
                    author=art_data.author,
                    summary=art_data.summary,
                    content=art_data.content,
                    source_type=art_data.source_type,
                    published_at=art_data.published_at,
                    article_metadata=art_data.article_metadata
                )
                db.add(db_article)
                inserted_count += 1
            
            db.commit()
        except Exception as e:
            print(f"Error executing scraper {scraper.name}: {str(e)}")
            db.rollback()

    return IngestResponse(
        status="completed",
        scraped_count=scraped_total,
        inserted_count=inserted_count,
        skipped_count=skipped_count
    )

@app.post("/process", response_model=ProcessResponse)
def trigger_process(db: Session = Depends(get_db)):
    """
    Processes newly ingested articles:
    1. Generates text embeddings and runs pgvector semantic deduplication.
    2. Runs LLM-based Scoring & Ranking on active (non-duplicate) items.
    """
    # 1. Semantic Deduplication
    try:
        duplicates_found = dedup_service.process_unprocessed_articles(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deduplication processing failed: {str(e)}")

    # 2. LLM-based Scoring/Ranking
    try:
        ranked_count = ranking_service.rank_unranked_articles(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking evaluation failed: {str(e)}")

    # Get total unprocessed count left (if any errored out)
    unprocessed_count = db.query(Article).filter(Article.is_duplicate == False, Article.overall_score.is_(None)).count()

    return ProcessResponse(
        status="completed",
        processed_count=duplicates_found + ranked_count,
        duplicates_found=duplicates_found,
        ranked_count=ranked_count
    )

@app.post("/generate-briefing")
def trigger_generate_briefing(
    target_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (defaults to current local date)"),
    db: Session = Depends(get_db)
):
    """
    Compiles daily briefings, generates personalized insights using the LLM,
    constructs the briefing document, and saves it.
    """
    if target_date:
        try:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    else:
        parsed_date = datetime.utcnow().date()

    try:
        briefing = insights_service.generate_daily_briefing(db, parsed_date)
        import os
        return {
            "status": "completed",
            "briefing_id": briefing.id,
            "date": briefing.date,
            "content": briefing.content,
            "smtp_sender": os.getenv("SMTP_SENDER"),
            "smtp_receiver": os.getenv("SMTP_RECEIVER")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate daily briefing: {str(e)}")

@app.get("/briefings/{target_date}", response_model=BriefingResponse)
def get_briefing(target_date: str, db: Session = Depends(get_db)):
    """
    Retrieves the compiled briefing for a specific date (YYYY-MM-DD).
    """
    try:
        parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    briefing = db.query(Briefing).filter(Briefing.date == parsed_date).first()
    if not briefing:
        raise HTTPException(status_code=404, detail=f"Briefing not found for date {target_date}")
    
    return briefing

@app.get("/search", response_model=List[ArticleResponse])
def semantic_search(
    query: str = Query(..., description="The query to search semantically (e.g. 'MCP local servers')"),
    limit: int = Query(10, description="Max number of items to return"),
    db: Session = Depends(get_db)
):
    """
    Performs a semantic vector search using pgvector across all ranked, non-duplicate articles.
    """
    try:
        # Generate query embedding
        query_vector = embedding_service.get_embedding(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {str(e)}")

    # pgvector distance operator
    distance_expr = Article.embedding.cosine_distance(query_vector)

    # Fetch nearest articles
    results = (
        db.query(Article)
        .filter(
            Article.is_duplicate == False,
            Article.embedding.is_not(None)
        )
        .order_by(distance_expr)
        .limit(limit)
        .all()
    )

    return results

@app.post("/interests")
def add_user_interest(
    name: str = Query(..., description="Interest name (e.g. 'MCP')"),
    tier: int = Query(..., description="Interest priority tier (1, 2, or 3)"),
    db: Session = Depends(get_db)
):
    """
    Adds a custom weighted user interest or updates an existing interest's tier.
    """
    if tier not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Tier must be 1, 2, or 3.")

    # Defaults weights: Tier 1 = 1.0, Tier 2 = 0.7, Tier 3 = 0.4
    weight = 1.0 if tier == 1 else (0.7 if tier == 2 else 0.4)

    existing = db.query(UserInterest).filter(UserInterest.interest_name == name).first()
    if existing:
        existing.tier = tier
        existing.weight = weight
        db.commit()
        return {"status": "success", "message": f"Updated interest '{name}' to Tier {tier}."}
    
    new_interest = UserInterest(interest_name=name, tier=tier, weight=weight)
    db.add(new_interest)
    db.commit()
    return {"status": "success", "message": f"Added interest '{name}' in Tier {tier}."}

@app.get("/interests")
def get_user_interests(db: Session = Depends(get_db)):
    """
    Lists all saved user interest categories and their prioritization tiers.
    """
    return db.query(UserInterest).order_by(UserInterest.tier.asc()).all()
