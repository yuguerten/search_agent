from types import SimpleNamespace

import numpy as np

from app.visualization.embeddings import project_embeddings, save_embedding_plot


def paper(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        embedding=[float(index), float(index + 1), float(index + 2)],
        final_score=index / 10,
        title=f"Paper {index}",
    )


def test_project_embeddings_returns_two_coordinates() -> None:
    coordinates = project_embeddings([paper(1), paper(2), paper(3)])

    assert coordinates.shape == (3, 2)
    assert np.isfinite(coordinates).all()


def test_save_embedding_plot_writes_png(tmp_path) -> None:
    output = save_embedding_plot(
        [paper(1), paper(2), paper(3)],
        tmp_path / "embeddings.png",
    )

    assert output.exists()
    assert output.suffix == ".png"
    assert output.read_bytes().startswith(b"\x89PNG")
