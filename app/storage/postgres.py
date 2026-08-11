from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings
from app.models import PaperCandidate


class Base(DeclarativeBase):
    pass


class PaperRecord(Base):
    """PostgreSQL representation of a paper and its ranking snapshot."""

    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[list[str]] = mapped_column(JSONB, default=list)
    arxiv_url: Mapped[str] = mapped_column(Text)
    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[date] = mapped_column(Date)
    updated_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(512), nullable=True)
    semantic_scholar_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    influential_citation_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    citation_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    citation_retrieved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    final_score: Mapped[float] = mapped_column(default=0.0)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PostgresPaperStore:
    """Persistence adapter used by tools, not by LLM prompts directly."""

    def __init__(self, database_url: str | None = None):
        settings = get_settings()
        self.engine = create_async_engine(database_url or settings.database_url)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def upsert_papers(
        self,
        papers: Sequence[PaperCandidate],
        embeddings: Sequence[list[float]] | None = None,
    ) -> None:
        async with self.sessions() as session:
            for index, paper in enumerate(papers):
                record = PaperRecord(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=paper.authors,
                    arxiv_url=paper.arxiv_url,
                    pdf_url=paper.pdf_url,
                    published_at=paper.published_at,
                    updated_at=paper.updated_at,
                    doi=paper.doi,
                    semantic_scholar_id=paper.semantic_scholar_id,
                    citation_count=paper.citation_count,
                    influential_citation_count=paper.influential_citation_count,
                    citation_source=paper.citation_source,
                    citation_retrieved_at=paper.citation_retrieved_at,
                    final_score=paper.final_score,
                    metadata_json=paper.model_dump(mode="json"),
                    embedding=embeddings[index] if embeddings else None,
                )
                await session.merge(record)
            await session.commit()

    async def close(self) -> None:
        await self.engine.dispose()
