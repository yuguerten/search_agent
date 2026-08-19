# Agentic Literature Researcher

A Google ADK and LiteLLM research workflow that asks clarification questions,
searches recent arXiv papers, enriches them with Semantic Scholar citations,
stores metadata and embeddings in PostgreSQL/pgvector, and produces a cited
five-paper synthesis.

## Architecture

```text
User
  -> Clarifier (one question per turn)
  -> Dispatcher (structured intent and search queries)
  -> Researcher -> Critic (maximum 3 iterations)
  -> Dispatcher -> Synthesizer -> Report
```

The researcher applies the strict rolling two-year filter before ranking. The
ranking combines relevance, age-adjusted citation impact, freshness, and metadata
quality. The critic performs deterministic checks before its semantic review.

## Setup

```bash
uv sync
cp .env.example .env
docker compose up -d postgres
```

The default LLM provider is OpenRouter with Nemotron. Export the key before
starting ADK, or place it in `.env`:

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key"
uv run adk web
```

Embeddings remain configured independently through LM Studio by default. Set
`EMBEDDING_PROVIDER` and its API settings if you want to move embeddings too.
Set `SEMANTIC_SCHOLAR_API_KEY` when available; the API can be used without a key
at lower rate limits.

Run the ADK playground from the repository root:

```bash
uv run adk web
```

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The external API and PostgreSQL tools are isolated from the unit tests. Add
integration tests with mocked HTTP responses before enabling live end-to-end runs.

### PostgreSQL / pgvector

The database must have the pgvector server extension installed. The application enables it automatically with `CREATE EXTENSION IF NOT EXISTS vector` before creating the papers table. If the database user cannot create extensions, run this once as a PostgreSQL administrator:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

