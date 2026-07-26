from fastapi import FastAPI
from pydantic import BaseModel

from embed import embed
from store import VectorStore
from similarity import cosine_similarity

app = FastAPI()

store = VectorStore()
category_vectors = store.get_vectors()


class Request(BaseModel):
    message: str


@app.post("/route")
def route(req: Request):
    query = req.message

    query_vec = embed(query)

    scores = {
        k: cosine_similarity(query_vec, v)
        for k, v in category_vectors.items()
    }

    best = max(scores, key=scores.get)

    return {
        "route": best,
        "scores": scores
    }