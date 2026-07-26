import json
from embed import embed

class VectorStore:
    def __init__(self, path="data/categories.json"):
        with open(path) as f:
            self.raw = json.load(f)

        self.vectors = {}
        self._build()

    def _build(self):
        for k, examples in self.raw.items():
            vecs = [embed(e) for e in examples]

            # 平均向量
            avg = [sum(x)/len(x) for x in zip(*vecs)]
            self.vectors[k] = avg

    def get_vectors(self):
        return self.vectors