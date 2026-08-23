from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sqlalchemy import select

from app.storage.postgres import PaperRecord, PostgresPaperStore


def project_embeddings(papers: Sequence[PaperRecord]) -> np.ndarray:
    """Project stored vectors to two dimensions with a deterministic PCA."""

    if not papers:
        raise ValueError("No papers with stored embeddings were found")

    vectors = np.asarray([paper.embedding for paper in papers], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] < 2:
        raise ValueError("Stored embeddings must contain at least two dimensions")
    if len(vectors) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    return PCA(n_components=2, random_state=42).fit_transform(vectors)


async def load_embedded_papers(database_url: str | None = None) -> list[PaperRecord]:
    """Load papers and vectors directly from PostgreSQL."""

    store = PostgresPaperStore(database_url)
    try:
        async with store.sessions() as session:
            result = await session.execute(
                select(PaperRecord)
                .where(PaperRecord.embedding.is_not(None))
                .order_by(PaperRecord.final_score.desc(), PaperRecord.arxiv_id)
            )
            return list(result.scalars().all())
    finally:
        await store.close()


def save_embedding_plot(
    papers: Sequence[PaperRecord],
    output_path: str | Path,
) -> Path:
    """Save a labelled PNG of the stored paper embeddings."""

    if not papers:
        raise ValueError("No papers with stored embeddings were found")

    coordinates = project_embeddings(papers)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray([paper.final_score for paper in papers], dtype=float)

    figure, axis = plt.subplots(figsize=(16, 10), constrained_layout=True)
    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=scores,
        cmap="viridis",
        s=180,
        edgecolors="white",
        linewidths=1.2,
    )
    for index, paper in enumerate(papers, start=1):
        title = " ".join(paper.title.split())
        label = f"{index}. {title[:72]}{'…' if len(title) > 72 else ''}"
        axis.annotate(
            label,
            (coordinates[index - 1, 0], coordinates[index - 1, 1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )

    figure.colorbar(points, ax=axis, label="Final ranking score")
    axis.set_title("Saved Research Paper Embeddings (PCA projection)")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.grid(alpha=0.2)
    figure.savefig(output, dpi=200, format="png", bbox_inches="tight")
    plt.close(figure)
    return output
