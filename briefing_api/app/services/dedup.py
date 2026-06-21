import logging
from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from app.models import Article
from app.services.embedding import embedding_service

logger = logging.getLogger("briefing_api.services.dedup")

# Similarity threshold: distance < 0.15 corresponds to similarity > 85%
DEDUP_DISTANCE_THRESHOLD = 0.15

class DeduplicationService:
    def deduplicate_article(self, db: Session, article: Article) -> bool:
        """
        Deduplicates an article against other articles from the last 48 hours.
        Returns True if the article is marked as a duplicate.
        """
        # If it already has an embedding, use it; otherwise generate it
        if article.embedding is None:
            try:
                # Combine title and summary for embedding context
                embed_text = f"{article.title}. {article.summary or ''}"
                article.embedding = embedding_service.get_embedding(embed_text)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to generate embedding for article {article.id}: {str(e)}")
                return False

        # Query database for articles from the last 48 hours (excluding itself and other duplicates)
        limit_date = datetime.utcnow() - timedelta(hours=48)
        
        # pgvector cosine distance operator is `<=>`
        # In SQLAlchemy, we use the .cosine_distance() method on the Vector column
        distance_expr = Article.embedding.cosine_distance(article.embedding)

        # Query for the closest candidate
        closest_candidate = (
            db.query(Article)
            .filter(
                Article.id != article.id,
                Article.is_duplicate == False,
                Article.published_at >= limit_date,
                Article.embedding.is_not(None)
            )
            .order_by(distance_expr)
            .first()
        )

        if closest_candidate:
            # Fetch the actual distance value
            # Since .first() doesn't return the distance unless requested, let's query the specific distance
            distance_val = db.query(distance_expr).filter(Article.id == closest_candidate.id).scalar()
            
            if distance_val is not None and distance_val < DEDUP_DISTANCE_THRESHOLD:
                logger.info(
                    f"Semantic duplicate found! \n"
                    f"  New: '{article.title}' ({article.url}) \n"
                    f"  Existing: '{closest_candidate.title}' ({closest_candidate.url}) \n"
                    f"  Cosine Distance: {distance_val:.4f} (Similarity: {(1 - distance_val)*100:.2f}%)"
                )
                article.is_duplicate = True
                article.duplicate_of_id = closest_candidate.id
                db.commit()
                return True

        return False

    def process_unprocessed_articles(self, db: Session) -> int:
        """
        Finds all active, non-processed articles and runs semantic deduplication.
        Returns the number of duplicates found.
        """
        unprocessed = (
            db.query(Article)
            .filter(
                Article.is_duplicate == False,
                Article.embedding.is_(None)
            )
            .all()
        )
        
        logger.info(f"Deduplication: Processing {len(unprocessed)} new articles...")
        duplicates_count = 0
        
        for article in unprocessed:
            if self.deduplicate_article(db, article):
                duplicates_count += 1
                
        return duplicates_count

dedup_service = DeduplicationService()
