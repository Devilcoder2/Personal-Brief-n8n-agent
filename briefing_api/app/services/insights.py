import logging
from datetime import datetime, date, timedelta
from typing import Optional, List
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

    def generate_daily_briefing(self, db: Session, target_date: date, source: Optional[str] = None) -> tuple[Briefing, str, str]:
        """
        Gathers articles/videos from the last 24 hours, filters and groups them,
        calls Ollama to generate personalized insights, structures the markdown and HTML,
        and saves the briefing to the database.
        """
        if not source:
            source = "youtube"

        start_time = datetime.combine(target_date, datetime.min.time()) - timedelta(hours=2)
        end_time = datetime.combine(target_date, datetime.max.time()) + timedelta(hours=2)

        # Get all non-duplicate articles from today, sorted by overall_score DESC (highest value first) and published_at DESC (newest first)
        articles = (
            db.query(Article)
            .filter(
                Article.is_duplicate == False,
                Article.overall_score.is_not(None),
                Article.source_type == source,
                Article.created_at >= start_time,
                Article.created_at <= end_time
            )
            .order_by(Article.overall_score.desc(), Article.published_at.desc())
            .all()
        )

        source_display = source.upper()

        if not articles:
            logger.warning(f"No articles found to generate briefing for date {target_date} from source {source}")
            empty_content = (
                f"# Daily Developer {source_display} Briefing\n"
                f"*Date: {target_date}*\n\n"
                f"No new developer updates were captured for today."
            )
            empty_fragment = f'<p style="color:#64748b; font-style:italic; margin-bottom:20px;">No new developer updates from {source_display} were captured for today.</p>'
            empty_html = self._compile_html(target_date, empty_fragment, source=source)
            
            # Save empty briefing
            existing_briefing = db.query(Briefing).filter(Briefing.date == target_date, Briefing.source_type == source).first()
            if existing_briefing:
                existing_briefing.content = empty_content
                db.commit()
                return existing_briefing, empty_html, empty_fragment
            else:
                briefing = Briefing(date=target_date, source_type=source, content=empty_content)
                db.add(briefing)
                db.commit()
                return briefing, empty_html, empty_fragment

        # Must Watch: Top 1 item (highest overall score)
        must_watch = articles[:1]
        must_watch_ids = {a.id for a in must_watch}

        # Rest of the items: Next 2 items (overall top 3 total)
        other_items = [a for a in articles if a.id not in must_watch_ids][:2]
        
        # Combine top 3 for the reference catalog
        top_three_items = must_watch + other_items

        # Call LLM to generate Personalized Insights based on top item descriptions
        insights_text = self._generate_personalized_insights_llm(must_watch, other_items, source=source)

        # Assemble Markdown Briefing
        markdown = self._compile_markdown(
            target_date=target_date,
            must_watch=must_watch,
            other_videos=other_items,
            insights=insights_text,
            all_articles=top_three_items,
            source=source
        )

        # Parse insights markdown to HTML
        insights_html = self._markdown_to_html(insights_text)

        # Compile HTML Fragment
        fragment_html = self._compile_html_fragment(
            target_date=target_date,
            must_watch=must_watch,
            other_videos=other_items,
            insights_html=insights_html,
            all_articles=top_three_items,
            source=source
        )

        # Assemble HTML Briefing
        html = self._compile_html(
            target_date=target_date,
            fragment_html=fragment_html,
            source=source
        )

        # Save to DB (override if briefing already exists for this date/source)
        existing_briefing = db.query(Briefing).filter(Briefing.date == target_date, Briefing.source_type == source).first()
        if existing_briefing:
            existing_briefing.content = markdown
            db.commit()
            return existing_briefing, html, fragment_html
        else:
            briefing = Briefing(date=target_date, source_type=source, content=markdown)
            db.add(briefing)
            db.commit()
            return briefing, html, fragment_html

    def _generate_personalized_insights_llm(self, must_watch: list[Article], other_videos: list[Article], source: str = "youtube") -> str:
        """
        Invokes Ollama to synthesize the daily summaries and write custom developer notes.
        """
        source_label = "items"
        source_category = "resources"
        recommend_verb = "reading"
        
        if source == "youtube":
            source_label = "videos"
            source_category = "developer channels"
            recommend_verb = "watching"
        elif source == "hn":
            source_label = "stories"
            source_category = "Hacker News"
            recommend_verb = "reading"
        elif source == "reddit":
            source_label = "posts"
            source_category = "Reddit"
            recommend_verb = "reading"
        elif source == "github":
            source_label = "repositories"
            source_category = "GitHub"
            recommend_verb = "exploring"
        elif source == "arxiv":
            source_label = "papers"
            source_category = "arXiv"
            recommend_verb = "reading"
        elif source == "blog":
            source_label = "articles"
            source_category = "tech blogs"
            recommend_verb = "reading"

        context_lines = []
        context_lines.append(f"### TOP MUST WATCH/READ {source_label.upper()}")
        for i, a in enumerate(must_watch, 1):
            context_lines.append(f"{i}. {a.title} by {a.author}\n   Summary: {a.summary or 'N/A'}")
        
        if other_videos:
            context_lines.append(f"\n### OTHER RELEVANT {source_label.upper()}")
            for a in other_videos:
                context_lines.append(f"- {a.title} by {a.author}\n  Summary: {a.summary or 'N/A'}")

        top_developments_context = "\n".join(context_lines)

        system_prompt = (
            f"You are a Staff AI Architect and Tech Analyst. Your job is to synthesize today's major {source_label} "
            f"from {source_category} and provide actionable insights for an experienced software developer interested in "
            "Backend Engineering, System Design, LLMs, AI Agents, MCP (Model Context Protocol), RAG, and Developer Tools.\n\n"
            "Format your response as a professional analysis with the following headings:\n"
            "#### Key Takeaways & Trends\n"
            f"<Explain the core trends or announcements shown in today's {source_label}>\n\n"
            "#### Impact on AI & Backend Engineers\n"
            f"<How these concepts/launches affect design choices, development workflows, or architectural patterns>\n\n"
            "#### Watch Recommendation\n"
            f"<Recommend specifically which 1-2 {source_label} are worth {recommend_verb} immediately and which ones to skip. Be direct.>\n\n"
            "Keep your output clean, brief, and in markdown format."
        )

        user_prompt = (
            f"Here are the details of developer {source_label} uploaded in the last 24 hours:\n\n"
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
                "#### Key Takeaways & Trends\n"
                f"New technical updates and tutorials were published today on {source_category}.\n\n"
                "#### Impact on AI & Backend Engineers\n"
                "New capabilities make standard implementations faster to ship.\n\n"
                "#### Watch Recommendation\n"
                f"Inspect the catalog below for uploads matching your active research areas."
            )

    def _markdown_to_html(self, md_text: str) -> str:
        """
        Robustly converts the LLM's simple markdown structure (headers, bold text, lists) into HTML.
        """
        if not md_text:
            return ""
        import re
        
        html = md_text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Parse bold (**text**)
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
        # Parse italic (*text*)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Parse inline code (`code`)
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        
        # Parse bullet lists
        lines = html.split('\n')
        in_list = False
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('- ') or stripped.startswith('* '):
                content = stripped[2:]
                if not in_list:
                    new_lines.append('<ul class="insights-list">')
                    in_list = True
                new_lines.append(f'<li class="insights-list-item">{content}</li>')
            else:
                if in_list:
                    new_lines.append('</ul>')
                    in_list = False
                new_lines.append(line)
        if in_list:
            new_lines.append('</ul>')
        html = '\n'.join(new_lines)
        
        # Parse headers
        html = re.sub(r'^#### (.*)$', r'<h4 class="insights-sub-header">\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.*)$', r'<h3 class="insights-sub-header">\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*)$', r'<h2 class="insights-header">\1</h2>', html, flags=re.MULTILINE)
        
        # Parse paragraphs
        paragraphs = html.split('\n\n')
        for i, p in enumerate(paragraphs):
            p_stripped = p.strip()
            # Wrap standard text sections in paragraph blocks, but avoid wrapping HTML containers/lists/headers
            if p_stripped and not p_stripped.startswith('<h') and not p_stripped.startswith('<u') and not p_stripped.startswith('<l') and not p_stripped.startswith('</u') and not p_stripped.startswith('<div') and not p_stripped.startswith('</div'):
                paragraphs[i] = f'<p class="insights-p">{p_stripped}</p>'
        
        html = '\n'.join(paragraphs)
        return html

    def _compile_html_fragment(self, target_date: date, must_watch: list[Article], other_videos: list[Article], 
                               insights_html: str, all_articles: list[Article], source: str = "youtube") -> str:
        """
        Assembles a styled HTML block for a specific source to be embedded in the main briefing.
        """
        source_label = "Items"
        source_singular = "Item"
        action_label = "Read"
        
        if source == "youtube":
            source_label = "YouTube Videos"
            source_singular = "Video"
            action_label = "Watch Video"
        elif source == "hn":
            source_label = "Hacker News Stories"
            source_singular = "Story"
            action_label = "Read Story"
        elif source == "reddit":
            source_label = "Reddit Posts"
            source_singular = "Post"
            action_label = "Read Post"
        elif source == "github":
            source_label = "GitHub Repositories"
            source_singular = "Repo"
            action_label = "View Repo"
        elif source == "arxiv":
            source_label = "Research Papers"
            source_singular = "Paper"
            action_label = "Read Paper"
        elif source == "blog":
            source_label = "Blog Articles"
            source_singular = "Article"
            action_label = "Read Article"

        # Build Executive Synthesis card
        if insights_html:
            insights_html_block = f"""
            <div class="insights-card">
              <div class="insights-title">⭐ Executive AI Synthesis ({source_label})</div>
              {insights_html}
            </div>
            """
        else:
            insights_html_block = ""

        # Build Must Watch list
        must_watch_html_list = []
        if must_watch:
            for a in must_watch:
                summary = a.summary or "No summary available."
                must_watch_html_list.append(f"""
                <div class="must-watch-card">
                  <h3><a href="{a.url}" target="_blank">{a.title}</a></h3>
                  <div class="meta-row">
                    <span class="channel-badge">{a.author}</span>
                    <a class="btn-watch" href="{a.url}" target="_blank">{action_label} ➔</a>
                  </div>
                  <blockquote>{summary}</blockquote>
                </div>
                """)
            must_watch_html = "\n".join(must_watch_html_list)
        else:
            must_watch_html = f'<p style="color:#64748b; font-style:italic; margin-bottom:20px;">No high-priority {source_label.lower()} cataloged today.</p>'

        # Build Other Videos list
        other_videos_html_list = []
        if other_videos:
            for a in other_videos:
                desc = a.summary or "No description available"
                other_videos_html_list.append(f"""
                <li class="video-list-item">
                  <a class="video-title-link" href="{a.url}" target="_blank">{a.title}</a>
                  <span class="channel-badge-small">{a.author}</span>
                  <p class="desc">{desc}</p>
                </li>
                """)
            other_videos_html = "\n".join(other_videos_html_list)
        else:
            other_videos_html = f'<p style="color:#64748b; font-style:italic; padding:10px 0;">No other {source_label.lower()} uploads captured today.</p>'

        # Build Reference Catalogue rows
        reference_rows_list = []
        for a in all_articles:
            reference_rows_list.append(f"""
            <tr>
              <td><code>{a.author}</code></td>
              <td>{a.title}</td>
              <td><a class="ref-link" href="{a.url}" target="_blank">{source_singular} Link</a></td>
            </tr>
            """)
        reference_rows = "\n".join(reference_rows_list) if reference_rows_list else f'<tr><td colspan="3" style="text-align:center; color:#64748b;">No reference {source_label.lower()} available today.</td></tr>'

        fragment = f"""
        <div class="source-section" style="margin-top: 40px; margin-bottom: 40px;">
          <h2 style="font-size: 20px; font-weight: 800; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 0; margin-bottom: 20px; text-transform: uppercase; color: #1e1b4b; letter-spacing: 0.5px;">🌐 {source_label}</h2>
          
          {insights_html_block}
          
          <div class="section-header" style="font-size: 14px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 12px; margin-top: 20px;">
            🔥 Must Watch / Read
          </div>
          {must_watch_html}
          
          <div class="section-header" style="font-size: 14px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 12px; margin-top: 20px;">
            🎥 More Releases & Updates
          </div>
          <ul class="video-list">
            {other_videos_html}
          </ul>
          
          <div class="section-header" style="font-size: 14px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 12px; margin-top: 20px;">
            🔗 Reference Catalogue
          </div>
          <table class="ref-table">
            <thead>
              <tr>
                <th style="width: 25%;">Author/Source</th>
                <th>Title</th>
                <th style="width: 15%;">Link</th>
              </tr>
            </thead>
            <tbody>
              {reference_rows}
            </tbody>
          </table>
        </div>
        """
        return fragment

    def _compile_html(self, target_date: date, fragment_html: str, source: str = "youtube") -> str:
        """
        Assembles a premium, highly-styled, responsive light-mode HTML template for the email briefing.
        """
        formatted_date = target_date.strftime('%B %d, %Y')
        
        # Premium HTML Template
        email_template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily Developer Briefing</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #1e293b;
      background-color: #f8fafc;
      margin: 0;
      padding: 40px 10px;
      -webkit-font-smoothing: antialiased;
    }}
    .wrapper {{
      max-width: 640px;
      margin: 0 auto;
      background-color: #ffffff;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      border: 1px solid #e2e8f0;
    }}
    .top-gradient-bar {{
      height: 6px;
      background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    }}
    .header {{
      padding: 40px 30px 25px 30px;
      text-align: center;
      background-color: #ffffff;
    }}
    .header h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 800;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: #0f172a;
    }}
    .header p {{
      margin: 8px 0 0 0;
      font-size: 13.5px;
      color: #64748b;
      font-weight: 500;
    }}
    .date-badge {{
      display: inline-block;
      margin-top: 14px;
      background-color: #e0e7ff;
      color: #4338ca;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 14px;
      border-radius: 100px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .content {{
      padding: 10px 30px 35px 30px;
    }}
    .section-header {{
      color: #0f172a;
      font-size: 16px;
      font-weight: 800;
      margin-top: 35px;
      margin-bottom: 20px;
      padding-bottom: 8px;
      border-bottom: 2px solid #f1f5f9;
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    .insights-card {{
      background-color: #faf5ff;
      border-left: 4px solid #a855f7;
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 30px;
    }}
    .insights-title {{
      font-size: 12px;
      font-weight: 800;
      color: #7e22ce;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 16px;
    }}
    .insights-p {{
      font-size: 14px;
      color: #3b0764;
      margin-bottom: 14px;
      line-height: 1.6;
    }}
    .insights-sub-header {{
      font-size: 13.5px;
      font-weight: 700;
      color: #581c87;
      margin-top: 20px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .insights-list {{
      margin: 8px 0 16px 0;
      padding-left: 20px;
    }}
    .insights-list-item {{
      font-size: 14px;
      color: #3b0764;
      margin-bottom: 6px;
      line-height: 1.55;
    }}
    .must-watch-card {{
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #f59e0b;
      border-radius: 10px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    }}
    .must-watch-card h3 {{
      margin: 0 0 12px 0;
      font-size: 17px;
      font-weight: 700;
      line-height: 1.45;
    }}
    .must-watch-card h3 a {{
      color: #0f172a;
      text-decoration: none;
    }}
    .must-watch-card h3 a:hover {{
      color: #4f46e5;
    }}
    .meta-row {{
      margin-bottom: 16px;
    }}
    .channel-badge {{
      display: inline-block;
      background-color: #fef3c7;
      color: #d97706;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-right: 12px;
    }}
    .btn-watch {{
      display: inline-block;
      color: #4f46e5;
      text-decoration: none;
      font-weight: 700;
      font-size: 13px;
    }}
    .btn-watch:hover {{
      text-decoration: underline;
    }}
    .must-watch-card blockquote {{
      margin: 16px 0 0 0;
      padding-left: 14px;
      border-left: 2px solid #e2e8f0;
      font-style: normal;
      font-size: 13.5px;
      color: #475569;
      line-height: 1.6;
    }}
    .video-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .video-list-item {{
      border-bottom: 1px solid #f1f5f9;
      padding: 16px 0;
    }}
    .video-list-item:last-child {{
      border-bottom: none;
      padding-bottom: 0;
    }}
    .video-list-item:first-child {{
      padding-top: 0;
    }}
    .video-title-link {{
      color: #0f172a;
      font-weight: 600;
      font-size: 15px;
      text-decoration: none;
      line-height: 1.4;
    }}
    .video-title-link:hover {{
      color: #4f46e5;
    }}
    .channel-badge-small {{
      display: inline-block;
      background-color: #f1f5f9;
      color: #475569;
      font-size: 10px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-left: 8px;
      vertical-align: middle;
    }}
    .video-list-item .desc {{
      margin: 6px 0 0 0;
      font-size: 13.5px;
      color: #64748b;
      line-height: 1.5;
    }}
    code {{
      font-family: Menlo, Consolas, Monaco, monospace;
      background-color: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 12px;
      color: #0f172a;
    }}
    .ref-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 15px;
      font-size: 12px;
    }}
    .ref-table th, .ref-table td {{
      border: 1px solid #e2e8f0;
      padding: 10px 12px;
      text-align: left;
    }}
    .ref-table th {{
      background-color: #f8fafc;
      color: #475569;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
    }}
    .ref-table tr:nth-child(even) {{
      background-color: #f8fafc;
    }}
    .ref-link {{
      color: #4f46e5;
      text-decoration: none;
      font-weight: 600;
    }}
    .ref-link:hover {{
      text-decoration: underline;
    }}
    .footer {{
      background-color: #f8fafc;
      color: #64748b;
      padding: 30px;
      text-align: center;
      font-size: 12px;
      border-top: 1px solid #e2e8f0;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="top-gradient-bar"></div>
    <div class="header">
      <h1>Daily Developer Briefing</h1>
      <p>Self-Hosted AI Tech Analyst & Executive Briefing Summary</p>
      <div class="date-badge">{formatted_date}</div>
    </div>
    <div class="content">
      {fragment_html}
    </div>
    <div class="footer">
      Generated automatically via Docker, pgvector, Ollama & n8n.<br>
      Zero recurring SaaS charges. Keep building.
    </div>
  </div>
</body>
</html>"""
        return email_template

    def _compile_markdown(self, target_date: date, must_watch: list[Article], other_videos: list[Article], 
                          insights: str, all_articles: list[Article], source: str = "youtube") -> str:
        """
        Combines sections into a single standard Markdown document.
        """
        source_label = "Items"
        source_singular = "Item"
        
        if source == "youtube":
            source_label = "Videos"
            source_singular = "Video"
        elif source == "hn":
            source_label = "Stories"
            source_singular = "Story"
        elif source == "reddit":
            source_label = "Posts"
            source_singular = "Post"
        elif source == "github":
            source_label = "Repositories"
            source_singular = "Repository"
        elif source == "arxiv":
            source_label = "Papers"
            source_singular = "Paper"
        elif source == "blog":
            source_label = "Articles"
            source_singular = "Article"

        md = []
        md.append(f"# Daily Developer {source_label} Briefing")
        md.append(f"**Date:** {target_date.strftime('%B %d, %Y')}")
        md.append("")
        md.append("---")
        md.append("")

        # Section 1: Must Watch/Read
        md.append(f"## 🔥 Must Watch / Read")
        if must_watch:
            for a in must_watch:
                md.append(f"### [{a.title}]({a.url})")
                md.append(f"**Author/Source:** `{a.author}` | **Link:** [View Resource]({a.url})")
                md.append(f"> {a.summary or 'No summary available.'}")
                md.append("")
        else:
            md.append(f"*No high-priority {source_label.lower()} cataloged today.*")
            md.append("")

        # Section 2: Other Items
        md.append(f"## 🎥 More Releases & Updates")
        if other_videos:
            for a in other_videos:
                md.append(f"- **[{a.title}]({a.url})** - `{a.author}`")
                md.append(f"  *Description: {a.summary or 'No description available'}*")
            md.append("")
        else:
            md.append(f"*No other {source_label.lower()} uploads captured today.*")
            md.append("")

        # Section 3: Personalized Insights
        md.append("## ⭐ Analyst Insights")
        md.append(insights)
        md.append("")

        # Section 4: References
        md.append(f"## 🔗 Sources & {source_label} Reference")
        md.append("| Source | Title | Link |")
        md.append("| --- | --- | --- |")
        for a in all_articles:
            md.append(f"| `{a.author}` | {a.title[:60]}... | [Link]({a.url}) |")
        md.append("")

        return "\n".join(md)

insights_service = InsightsService()

