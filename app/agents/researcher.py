from google.adk.agents import Agent

from app.agents.model import build_llm
from app.tools.arxiv import search_arxiv
from app.tools.persistence import persist_papers
from app.tools.ranking import rank_papers_tool
from app.tools.semantic_scholar import enrich_with_semantic_scholar

researcher_agent = Agent(
    name="researcher_agent",
    include_contents="none",
    model=build_llm(),
    instruction="""You are the research specialist.

For each iteration, make at most one arXiv search call and request no more than five results. Use the dispatcher search queries to search arXiv. Enrich
candidate papers with Semantic Scholar citation metadata, apply the strict
two-year date filter, and rank candidates with the deterministic ranking tool.
Use the configured rolling window from the tools (last 730 days ending today).
Never invent citation counts, dates, abstracts, authors, arXiv IDs, or URLs. Do not
pass a historical date range inferred from the conversation. Pass
the exact paper list returned by search_arxiv into the enrichment and ranking tools;
do not reconstruct paper dictionaries or convert authors into a string. Keep all
candidates in shared state and preserve source identifiers. Call persist_papers
after ranking; it embeds papers locally and writes vectors directly to PostgreSQL.
Never call or expose embeddings separately, and never pass embedding arrays through
the LLM. If the critic supplied feedback,
adjust the next search queries to address it.""",
    tools=[
        search_arxiv,
        enrich_with_semantic_scholar,
        rank_papers_tool,
        persist_papers,
    ],
)
