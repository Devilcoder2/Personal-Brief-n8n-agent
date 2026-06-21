# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from app.database import get_db, init_db, SessionLocal
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

# Dictionary to track background ranking statuses per source
ranking_statuses: Dict[str, str] = {}

# Background ranking task executor
def run_background_ranking(source: str, db_session_factory):
    global ranking_statuses
    ranking_statuses[source] = "processing"
    db = db_session_factory()
    try:
        # Perform semantic deduplication and source-specific LLM ranking
        dedup_service.process_unprocessed_articles(db)
        ranking_service.rank_source_articles(db, source)
        ranking_statuses[source] = "completed"
    except Exception as e:
        print(f"Error in background ranking task for source {source}: {str(e)}")
        ranking_statuses[source] = "failed"
    finally:
        db.close()

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

@app.post("/clear-db")
def trigger_clear_db(db: Session = Depends(get_db)):
    """
    Truncates the articles table to clear historical data.
    """
    # pyrefly: ignore [missing-import]
    from sqlalchemy import text
    try:
        db.execute(text("TRUNCATE articles CASCADE;"))
        db.commit()
        return {"status": "success", "message": "Articles table truncated successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database truncation failed: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
def trigger_ingest(
    source: Optional[str] = Query(None, description="Specific source to ingest (e.g. 'youtube', 'hn', 'reddit', 'github', 'arxiv', 'blog')"),
    db: Session = Depends(get_db)
):
    """
    Runs platform scrapers and inserts newly found tech news into the database.
    """
    scraped_total = 0
    inserted_count = 0
    skipped_count = 0

    for scraper in SCRAPERS:
        if source:
            source_lower = source.lower()
            name_map = {
                "youtube": "youtube",
                "hn": "hacker news",
                "hackernews": "hacker news",
                "reddit": "reddit",
                "github": "github",
                "arxiv": "arxiv",
                "paper": "arxiv",
                "research": "arxiv",
                "blog": "blogs",
                "blogs": "blogs"
            }
            mapped_target = name_map.get(source_lower, source_lower)
            if scraper.name.lower() != mapped_target:
                continue

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

@app.post("/rank")
def trigger_rank(
    source: str = Query(..., description="Source type to rank (e.g. 'youtube', 'hn', 'reddit', 'github', 'arxiv', 'blog')"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Starts LLM ranking for a specific source asynchronously in the background.
    """
    global ranking_statuses
    if ranking_statuses.get(source) == "processing":
        return {
            "status": "processing",
            "message": f"Ranking is already running for source: {source}"
        }
    
    ranking_statuses[source] = "processing"
    background_tasks.add_task(run_background_ranking, source, SessionLocal)
    return {
        "status": "processing",
        "message": f"LLM ranking for source '{source}' started in the background."
    }

@app.get("/rank-status")
def get_rank_status(
    source: str = Query(..., description="Source type to check status (e.g. 'youtube', 'hn')")
):
    """
    Checks the status of the background ranking task for a specific source.
    """
    global ranking_statuses
    status = ranking_statuses.get(source, "idle")
    return {
        "status": status,
        "source": source
    }

@app.post("/generate-briefing")
def trigger_generate_briefing(
    target_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (defaults to current local date)"),
    source: Optional[str] = Query(None, description="Specific source to compile briefing for (e.g. 'youtube', 'hn', 'reddit', 'github', 'arxiv', 'blog')"),
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
        briefing, html_content, fragment_content = insights_service.generate_daily_briefing(db, parsed_date, source=source)
        import os
        return {
            "status": "completed",
            "briefing_id": briefing.id,
            "date": briefing.date,
            "content": briefing.content,
            "html": html_content,
            "fragment": fragment_content,
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
