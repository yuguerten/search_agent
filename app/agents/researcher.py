from google.adk.agents import Agent

from app.agents.model import build_llm
from app.tools.arxiv import search_arxiv
from app.tools.embeddings import embed_papers_tool
from app.tools.persistence import persist_papers
from app.tools.ranking import rank_papers_tool
from app.tools.semantic_scholar import enrich_with_semantic_scholar

researcher_agent = Agent(
    name="researcher_agent",
    model=build_llm(),
    instruction="""You are the research specialist.

For each iteration, use the dispatcher search queries to search arXiv. Enrich
candidate papers with Semantic Scholar citation metadata, apply the strict
two-year date filter, and rank candidates with the deterministic ranking tool.
Never invent citation counts, dates, abstracts, or authors. Keep all candidates in
shared state and preserve source identifiers. Embed eligible papers and persist
them in PostgreSQL with pgvector after ranking. If the critic supplied feedback,
adjust the next search queries to address it.""",
    tools=[
        search_arxiv,
        enrich_with_semantic_scholar,
        rank_papers_tool,
        embed_papers_tool,
        persist_papers,
    ],
)
