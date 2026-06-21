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
        Calls Ollama to score an article's importance, relevance, and novelty.
        Saves scores to the database and returns True on success.
        """
        interests_context = self._get_interests_prompt_context(db)
        
        system_prompt = (
            "You are an expert AI Analyst and Tech Research Assistant. "
            "Your task is to analyze technical content (articles, videos, papers) and rate it based on user interests. "
            "You must output ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "importance": <integer 1-10>,\n'
            '  "relevance": <integer 1-10>,\n'
            '  "novelty": <integer 1-10>,\n'
            '  "reasoning": "<short sentence explaining scores>"\n'
            "}\n"
            "Do not include any explanation outside the JSON object."
        )

        user_prompt = (
            f"Please evaluate this article according to the user's tiered interests.\n\n"
            f"--- User Interests ---\n"
            f"{interests_context}\n"
            f"--- Scoring Criteria ---\n"
            f"1. Importance (1-10): Technical depth, impact on the industry, or architectural significance.\n"
            f"2. Relevance (1-10): How closely it matches the user interests list. Higher tiers MUST get higher relevance scores (e.g. Tier 1 matches = 9-10, Tier 2 = 7-8, Tier 3 = 4-6, unrelated = 1).\n"
            f"3. Novelty (1-10): How unique, new, or non-repetitive the information is.\n\n"
            f"--- Article Details ---\n"
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
                
                importance = float(scores.get("importance", 5))
                relevance = float(scores.get("relevance", 5))
                novelty = float(scores.get("novelty", 5))
                
                # Compute composite score
                # Overall Score = (Relevance * 0.5) + (Importance * 0.3) + (Novelty * 0.2)
                overall_score = (relevance * 0.5) + (importance * 0.3) + (novelty * 0.2)
                
                article.importance_score = importance
                article.relevance_score = relevance
                article.novelty_score = novelty
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
            article.novelty_score = 5.0
            article.overall_score = 5.0
            db.commit()
            return False

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
