import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from sd2vec import SDVectors


class HybridVectors:
    def __init__(self, sd_vectors: SDVectors, w2v_vectors: KeyedVectors):
        self.sd_vectors = sd_vectors
        self.w2v_vectors = w2v_vectors
        self.sd_size = sd_vectors.vector_size
        self.w2v_size = w2v_vectors.vector_size
        self.vector_size = self.sd_size + self.w2v_size

        sd_vocab = set(sd_vectors.index_to_key)
        w2v_vocab = set(w2v_vectors.index_to_key)
        self.index_to_key = sorted(sd_vocab & w2v_vocab)
        self.key_to_index = {k: i for i, k in enumerate(self.index_to_key)}

        self.vectors = self._build_hybrid_vectors()

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _build_hybrid_vectors(self) -> np.ndarray:
        n = len(self.index_to_key)
        hybrid = np.zeros((n, self.vector_size), dtype=np.float32)

        for i, word in enumerate(self.index_to_key):
            sd_vec = self._normalize(self.sd_vectors.get_vector(word))
            w2v_vec = self._normalize(self.w2v_vectors[word])
            hybrid[i] = np.concatenate([sd_vec, w2v_vec])

        return hybrid

    def get_vector(self, word: str) -> np.ndarray:
        if word not in self.key_to_index:
            raise KeyError(f"'{word}' not in vocabulary")
        return self.vectors[self.key_to_index[word]]

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        if word not in self.key_to_index:
            raise KeyError(f"'{word}' not in vocabulary")

        query = self.get_vector(word)
        norms = np.linalg.norm(self.vectors, axis=1)
        scores = self.vectors @ query / (norms * np.linalg.norm(query) + 1e-8)

        excluded = {word}
        results = [
            (self.index_to_key[i], float(scores[i]))
            for i in np.argsort(-scores)
            if self.index_to_key[i] not in excluded
        ][:topn]
        return results

    def save_npy(self, path: str) -> None:
        np.save(path, self.vectors)

    def save_csv(self, path: str) -> None:
        import csv
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"dim_{i}" for i in range(self.vector_size)])
            for i, word in enumerate(self.index_to_key):
                writer.writerow([word] + self.vectors[i].tolist())

    @classmethod
    def load(cls, sd_path: str, w2v_path: str) -> "HybridVectors":
        sd_vectors = SDVectors.load(sd_path)
        w2v_vectors = KeyedVectors.load(w2v_path)
        return cls(sd_vectors, w2v_vectors)
