import logging
# pyrefly: ignore [missing-import]
import httpx
from app.config import settings

logger = logging.getLogger("briefing_api.services.embedding")

class EmbeddingService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_EMBED_MODEL

    def get_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding for the given text using local Ollama.
        """
        if not text:
            # Fallback for empty text
            text = "empty"
        
        # Clean text: remove null bytes and excessive whitespace
        text = text.replace("\x00", "").strip()
        
        # Limit text length to avoid token limit errors in small embedding models
        # nomic-embed-text typically supports 8k context, but 4k is plenty for scraping metadata
        text_truncated = text[:4000]

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text_truncated
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 404:
                    # Try alternate embed endpoint /api/embed
                    url_alt = f"{self.base_url}/api/embed"
                    payload_alt = {
                        "model": self.model,
                        "input": text_truncated
                    }
                    response = client.post(url_alt, json=payload_alt)
                    response.raise_for_status()
                    data = response.json()
                    # /api/embed returns 'embeddings' list of lists
                    embeddings = data.get("embeddings", [])
                    if embeddings:
                        return embeddings[0]
                
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding via Ollama: {str(e)}")
            raise e
embedding_service = EmbeddingService()
