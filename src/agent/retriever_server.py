"""Persistent local dense retriever service for BGE-M3/NV-Embed."""
from __future__ import annotations

import argparse
import os

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.segmenter import _dense_embed


class TopKRequest(BaseModel):
    paragraphs: list[str]
    query: str
    candidate_ids: list[int] | None = None
    top_k: int = 8
    batch_size: int = 16
    max_length: int | None = None


app = FastAPI(title="Film-CS Dense Retriever")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/topk")
def topk(req: TopKRequest) -> dict[str, object]:
    candidate_ids = req.candidate_ids
    if candidate_ids is None:
        candidate_ids = list(range(len(req.paragraphs)))
    candidate_ids = [i for i in candidate_ids if 0 <= i < len(req.paragraphs)]
    if not candidate_ids or req.top_k <= 0:
        return {"ids": [], "scores": []}

    batch_size = int(os.environ.get("DENSE_BATCH_SIZE", str(req.batch_size)))
    max_length = int(os.environ.get("DENSE_MAX_LENGTH", str(req.max_length or 512)))
    query_emb = _dense_embed([req.query], batch_size=1, max_length=max_length, is_query=True)
    doc_emb = _dense_embed(
        [req.paragraphs[i] for i in candidate_ids],
        batch_size=batch_size,
        max_length=max_length,
        is_query=False,
    )
    scores = doc_emb @ query_emb[0]
    order = np.argsort(-scores)[:req.top_k]
    ids = [candidate_ids[int(i)] for i in order]
    return {"ids": ids, "scores": [float(scores[int(i)]) for i in order]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    os.environ.pop("DENSE_RETRIEVER_URL", None)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
