import json
import logging
# pyrefly: ignore [missing-import]
import httpx
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Article, UserInterest

logger = logging.getLogger("briefing_api.services.ranker")

# Default interests fallback if user_interests table is empty
DEFAULT_INTERESTS = {
    1: ["Backend Engineering", "System Design", "LLMs", "AI Agents", "MCP", "RAG", "AI Engineering", "AI Infrastructure", "Realtime AI", "Voice AI"],
    2: ["Vector Databases", "Redis", "Semantic Search", "Open Source AI", "Developer Tools", "AI Research Papers", "AI Product Launches"],
    3: ["Frontend Engineering", "Mapping Systems", "Startups"]
}

DEFAULT_WEIGHTS = {
    1: 1.0,  # Tier 1
    2: 0.7,  # Tier 2
    3: 0.4   # Tier 3
}

class RankingService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_LLM_MODEL

    def _get_interests_prompt_context(self, db: Session) -> str:
        """
        Builds a text description of user interests and tiers for the LLM prompt.
        """
        interests = db.query(UserInterest).all()
        if not interests:
            # Use defaults
            t1 = ", ".join(DEFAULT_INTERESTS[1])
            t2 = ", ".join(DEFAULT_INTERESTS[2])
            t3 = ", ".join(DEFAULT_INTERESTS[3])
        else:
            t1_list = [i.interest_name for i in interests if i.tier == 1]
            t2_list = [i.interest_name for i in interests if i.tier == 2]
            t3_list = [i.interest_name for i in interests if i.tier == 3]
            t1 = ", ".join(t1_list) if t1_list else "None"
            t2 = ", ".join(t2_list) if t2_list else "None"
            t3 = ", ".join(t3_list) if t3_list else "None"

        return (
            f"Tier 1 (Highest Priority, Weight 1.0): {t1}\n"
            f"Tier 2 (High Priority, Weight 0.7): {t2}\n"
            f"Tier 3 (Medium Priority, Weight 0.4): {t3}\n"
        )

    def rank_article(self, db: Session, article: Article) -> bool:
        """
        Calls Ollama to score an article based on whether it deserves time or can be skipped
        without missing any technological advancements or new developer techniques.
        """
        interests_context = self._get_interests_prompt_context(db)
        
        system_prompt = (
            "You are an expert AI Analyst and Tech Research Assistant. "
            "Your task is to analyze technical content (articles, videos, papers) and rate its value to a professional developer. "
            "You must output ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "deserves_time": <integer 1-10>,\n'
            '  "tech_advancement": <integer 1-10>,\n'
            '  "reasoning": "<short sentence explaining scores>"\n'
            "}\n"
            "Do not include any explanation outside the JSON object."
        )

        user_prompt = (
            f"Please evaluate this content based on whether it deserves a developer's time or if it can be skipped with zero effect on knowledge, advancement, or learning new technology techniques.\n\n"
            f"--- Scoring Criteria ---\n"
            f"1. Deserves Time (1-10): Score 9-10 if it offers high-value knowledge, deep technical insights, or actionable patterns. Score 1-3 if it is clickbait, repetitive, a basic surface-level summary, or safe to skip with no loss of knowledge.\n"
            f"2. Tech Advancement (1-10): Score 9-10 if it introduces a major technological advancement, new framework release, innovative architectural pattern, or breakthrough tool. Score 1-3 if it covers standard tools/concepts with no new advancements.\n\n"
            f"--- Content Details ---\n"
            f"Title: {article.title}\n"
            f"Source: {article.source_type.upper()} ({article.author or 'Unknown'})\n"
            f"Description/Summary: {article.summary or 'No description available'}\n"
        )

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1 # Low temperature for consistent scoring
            }
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                response_text = result.get("response", "").strip()
                scores = json.loads(response_text)
                
                deserves_time = float(scores.get("deserves_time", 5))
                tech_advancement = float(scores.get("tech_advancement", 5))
                
                # Compute composite score:
                # Overall Score = (deserves_time * 0.6) + (tech_advancement * 0.4)
                overall_score = (deserves_time * 0.6) + (tech_advancement * 0.4)
                
                # Map to existing db columns to avoid migrations
                article.importance_score = deserves_time
                article.relevance_score = tech_advancement
                article.novelty_score = 0.0
                article.overall_score = round(overall_score, 2)
                
                # Update reasoning in metadata
                meta = dict(article.article_metadata or {})
                meta["ranking_reasoning"] = scores.get("reasoning", "")
                article.article_metadata = meta
                
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to score article '{article.title}' using LLM: {str(e)}")
            # Set default safe fallback scores so pipeline does not halt
            article.importance_score = 5.0
            article.relevance_score = 5.0
            article.novelty_score = 0.0
            article.overall_score = 5.0
            db.commit()
            return False

    def rank_source_articles(self, db: Session, source: str) -> int:
        """
        Scores all non-duplicate articles for a specific source from the last 36 hours that haven't been scored yet.
        Returns the number of articles scored.
        """
        from datetime import datetime, timedelta
        time_limit = datetime.utcnow() - timedelta(hours=36)

        unranked = (
            db.query(Article)
            .filter(
                Article.is_duplicate == False,
                Article.source_type == source,
                Article.overall_score.is_(None),
                Article.created_at >= time_limit
            )
            .all()
        )
        
        logger.info(f"Ranking: Evaluating {len(unranked)} articles for source {source} with LLM...")
        ranked_count = 0
        for article in unranked:
            if self.rank_article(db, article):
                ranked_count += 1
                
        return ranked_count

    def rank_unranked_articles(self, db: Session) -> int:
        """
        Scores all non-duplicate articles from the last 36 hours that haven't been scored yet.
        Returns the number of articles scored.
        """
        from datetime import datetime, timedelta
        time_limit = datetime.utcnow() - timedelta(hours=36)

        unranked = (
            db.query(Article)
            .filter(
                Article.is_duplicate == False,
                Article.overall_score.is_(None),
                Article.created_at >= time_limit
            )
            .all()
        )
        
        logger.info(f"Ranking: Evaluating {len(unranked)} articles with LLM...")
        ranked_count = 0
        for article in unranked:
            if self.rank_article(db, article):
                ranked_count += 1
                
        return ranked_count

ranking_service = RankingService()
