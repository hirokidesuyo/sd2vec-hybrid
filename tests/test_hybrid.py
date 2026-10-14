import numpy as np
import pytest
from sd2vec_hybrid import HybridVectors


class MockSDVectors:
    def __init__(self):
        self.vector_size = 2
        self.index_to_key = ["猫", "犬", "鳥"]
        self.key_to_index = {k: i for i, k in enumerate(self.index_to_key)}
        self.vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def get_vector(self, word):
        return self.vectors[self.key_to_index[word]]


class MockW2VVectors:
    def __init__(self):
        self.vector_size = 3
        self.index_to_key = ["猫", "犬", "魚"]
        self.key_to_index = {k: i for i, k in enumerate(self.index_to_key)}
        self.vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def __getitem__(self, word):
        return self.vectors[self.key_to_index[word]]


def test_hybrid_vectors():
    sd = MockSDVectors()
    w2v = MockW2VVectors()
    model = HybridVectors(sd, w2v)

    assert model.vector_size == 5
    assert set(model.index_to_key) == {"猫", "犬"}

    vec = model.get_vector("猫")
    assert vec.shape == (5,)
    assert np.allclose(np.linalg.norm(vec), 1.0)

    results = model.most_similar("猫", topn=1)
    assert results[0][0] == "犬"
