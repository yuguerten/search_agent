from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.visualization.embeddings import load_embedded_papers, save_embedding_plot


async def main(output_path: Path) -> None:
    papers = await load_embedded_papers()
    saved_path = save_embedding_plot(papers, output_path)
    print(f"Saved {len(papers)} paper embeddings to {saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot saved paper embeddings")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/paper_embeddings.png"),
        help="PNG output path",
    )
    args = parser.parse_args()
    asyncio.run(main(args.output))
