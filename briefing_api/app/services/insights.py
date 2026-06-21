import logging
from datetime import datetime, date, timedelta
# pyrefly: ignore [missing-import]
import httpx
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Article, Briefing

logger = logging.getLogger("briefing_api.services.insights")

class InsightsService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_LLM_MODEL

    def generate_daily_briefing(self, db: Session, target_date: date) -> Briefing:
        """
        Gathers articles from the last 24 hours, filters and groups them,
        calls Ollama to generate personalized insights, structures the markdown,
        and saves the briefing to the database.
        """
        # Fetch articles created in the last 24 hours (non-duplicate and ranked)
        start_time = datetime.combine(target_date, datetime.min.time()) - timedelta(hours=2) # Slight overlap to catch late night runs
        end_time = datetime.combine(target_date, datetime.max.time()) + timedelta(hours=2)

        articles = (
            db.query(Article)
            .filter(
                Article.is_duplicate == False,
                Article.overall_score.is_not(None),
                Article.created_at >= start_time,
                Article.created_at <= end_time
            )
            .order_by(Article.overall_score.desc())
            .all()
        )

        if not articles:
            logger.warning(f"No articles found to generate briefing for date {target_date}")
            empty_content = (
                f"# Daily AI & Tech Intelligence Briefing\n"
                f"*Date: {target_date}*\n\n"
                f"No new updates were found or ranked for today. All ingested items were filtered as noise."
            )
            briefing = Briefing(date=target_date, content=empty_content)
            db.add(briefing)
            db.commit()
            return briefing

        # 1. Group articles by section
        # Filter threshold: only keep items with score >= 5.0 to filter out noise
        filtered_articles = [a for a in articles if a.overall_score >= 5.0]
        
        # Must Reads: Top 3 overall scoring items
        must_reads = filtered_articles[:3]
        must_read_ids = {a.id for a in must_reads}

        # Rest of articles, categorized
        remaining = [a for a in filtered_articles if a.id not in must_read_ids]

        new_tools = [a for a in remaining if a.source_type == "github"][:4]
        papers = [a for a in remaining if a.source_type == "arxiv"][:4]
        videos = [a for a in remaining if a.source_type == "youtube"][:4]
        discussions = [a for a in remaining if a.source_type in ("hn", "reddit")][:5]

        # 2. Call LLM to generate Personalized Insights (Section 6)
        insights_text = self._generate_personalized_insights_llm(must_reads, new_tools, papers)

        # 3. Assemble Markdown Briefing
        markdown = self._compile_markdown(
            target_date=target_date,
            must_reads=must_reads,
            new_tools=new_tools,
            papers=papers,
            videos=videos,
            discussions=discussions,
            insights=insights_text,
            all_articles=filtered_articles
        )

        # 4. Save to DB (override if briefing already exists for this date)
        existing_briefing = db.query(Briefing).filter(Briefing.date == target_date).first()
        if existing_briefing:
            existing_briefing.content = markdown
            db.commit()
            return existing_briefing
        else:
            briefing = Briefing(date=target_date, content=markdown)
            db.add(briefing)
            db.commit()
            return briefing

    def _generate_personalized_insights_llm(self, must_reads: list[Article], new_tools: list[Article], papers: list[Article]) -> str:
        """
        Invokes Ollama to synthesize the top daily developments and write custom engineer insights.
        """
        # Build text description of top developments
        context_lines = []
        context_lines.append("### MUST READS")
        for i, a in enumerate(must_reads, 1):
            context_lines.append(f"{i}. {a.title} ({a.source_type.upper()}) - Score: {a.overall_score}\n   Summary: {a.summary or 'N/A'}")
        
        if new_tools:
            context_lines.append("\n### NEW TOOLS")
            for a in new_tools:
                context_lines.append(f"- {a.title} - Score: {a.overall_score}\n  Summary: {a.summary or 'N/A'}")
        
        if papers:
            context_lines.append("\n### RESEARCH PAPERS")
            for a in papers:
                context_lines.append(f"- {a.title} - Score: {a.overall_score}\n  Summary: {a.summary or 'N/A'}")

        top_developments_context = "\n".join(context_lines)

        system_prompt = (
            "You are a Staff AI Architect and Tech Analyst. Your job is to synthesize today's major tech news "
            "and provide actionable insights for an experienced software developer interested in Backend Engineering, "
            "System Design, LLMs, AI Agents, MCP (Model Context Protocol), RAG, and Vector Databases.\n\n"
            "Format your response as a professional analysis with the following headings:\n"
            "#### Why This Matters\n"
            "<Explain the core shift or significance of today's updates>\n\n"
            "#### Impact on AI & Backend Engineers\n"
            "<What architectural or design adjustments engineers should think about based on these updates>\n\n"
            "#### Reading Recommendation\n"
            "<Recommend specifically which 1-2 items from the list they should read immediately and which ones they can skip. Be direct.>\n\n"
            "Keep your output clean and concise, in markdown format."
        )

        user_prompt = (
            f"Here are the top technical developments from the last 24 hours:\n\n"
            f"{top_developments_context}\n\n"
            f"Please generate the personalized insights analysis."
        )

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Error compiling LLM insights: {str(e)}")
            return (
                "#### Why This Matters\n"
                "Significant updates in AI Agent architectures and developer tooling were surfaced today.\n\n"
                "#### Impact on AI & Backend Engineers\n"
                "System integration complexity continues to decrease as standards like MCP mature.\n\n"
                "#### Reading Recommendation\n"
                "Review the GitHub trending repositories to inspect new libraries and frameworks."
            )

    def _compile_markdown(self, target_date: date, must_reads: list[Article], new_tools: list[Article], 
                          papers: list[Article], videos: list[Article], discussions: list[Article], 
                          insights: str, all_articles: list[Article]) -> str:
        """
        Combines sections into a single standard Markdown document.
        """
        md = []
        md.append(f"# Daily AI & Tech Intelligence Briefing")
        md.append(f"**Date:** {target_date.strftime('%B %d, %Y')}")
        md.append("")
        md.append("---")
        md.append("")

        # Section 1: Must Read
        md.append("## 🔥 Must Read")
        if must_reads:
            for a in must_reads:
                reasoning = a.article_metadata.get("ranking_reasoning", "")
                md.append(f"### [{a.title}]({a.url})")
                md.append(f"**Source:** `{a.source_type.upper()}` | **Author/Channel:** {a.author or 'N/A'} | **Intelligence Score:** `{a.overall_score}/10` (R: {a.relevance_score}, I: {a.importance_score}, N: {a.novelty_score})")
                md.append(f"> {a.summary or 'No summary available.'}")
                if reasoning:
                    md.append(f"**Analyst Notes:** *{reasoning}*")
                md.append("")
        else:
            md.append("*No high-priority must reads found for today.*")
            md.append("")

        # Section 2: New Tools (GitHub)
        md.append("## 🚀 New Tools & Repositories")
        if new_tools:
            for a in new_tools:
                md.append(f"- **[{a.title}]({a.url})** - `{a.author}` (Stars: {a.article_metadata.get('stars', 'N/A')})")
                md.append(f"  *Description: {a.summary or 'N/A'}*")
            md.append("")
        else:
            md.append("*No new developer tools cataloged today.*")
            md.append("")

        # Section 3: Research Papers
        md.append("## 📄 Research Papers")
        if papers:
            for a in papers:
                md.append(f"- **[{a.title}]({a.url})** by {a.author}")
                md.append(f"  *Abstract: {a.summary or 'N/A'}*")
            md.append("")
        else:
            md.append("*No new research papers cataloged today.*")
            md.append("")

        # Section 4: Videos
        md.append("## 🎥 Videos")
        if videos:
            for a in videos:
                md.append(f"- **[{a.title}]({a.url})** - YouTube: `{a.author}`")
                md.append(f"  *Description: {a.summary or 'N/A'}*")
            md.append("")
        else:
            md.append("*No channel video uploads captured today.*")
            md.append("")

        # Section 5: Community Discussions
        md.append("## 💬 Community Discussions")
        if discussions:
            for a in discussions:
                score_label = "Upvotes" if a.source_type == "reddit" else "Points"
                score_val = a.article_metadata.get("upvotes") or a.article_metadata.get("points") or 0
                comments = a.article_metadata.get("comments_count") or 0
                source_label = f"r/{a.article_metadata.get('subreddit')}" if a.source_type == "reddit" else "Hacker News"
                md.append(f"- **[{a.title}]({a.url})** ({source_label} | {score_val} {score_label} | {comments} comments)")
            md.append("")
        else:
            md.append("*No significant developer community threads captured today.*")
            md.append("")

        # Section 6: Personalized Insights
        md.append("## ⭐ Personalized Insights")
        md.append(insights)
        md.append("")

        # Section 7: Source Links Reference
        md.append("## 🔗 Sources & Scores Reference")
        md.append("| Source Type | Score | Document Title | Link |")
        md.append("| --- | --- | --- | --- |")
        for a in all_articles:
            md.append(f"| `{a.source_type.upper()}` | `{a.overall_score}/10` | {a.title[:60]}... | [Link]({a.url}) |")
        md.append("")

        return "\n".join(md)

insights_service = InsightsService()
