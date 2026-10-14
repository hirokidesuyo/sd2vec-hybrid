import numpy as np
from sd2vec import SDVectors
from gensim.models import KeyedVectors
from hybrid_vectors import HybridVectors

sd = SDVectors.load(r'C:\Users\hi950\sd2vec\large_vocab_200000')
w2v = KeyedVectors.load(r'C:\Users\hi950\sd2vec-hybrid\data\chive-1.3-mc90_gensim\chive-1.3-mc90.kv')
hybrid = HybridVectors(sd, w2v)

np.random.seed(42)
sample_words = list(np.random.choice(hybrid.index_to_key, 100, replace=False))

def avg_diversity(model, words):
    all_neighbors = []
    for w in words:
        neighbors = [n for n, _ in model.most_similar(w, topn=10)]
        all_neighbors.extend(neighbors)
    unique = len(set(all_neighbors))
    total = len(all_neighbors)
    return unique / total

queries = ['猫', '春', '勝利', '日本', '愛']

print("=== Nearest Neighbors Comparison ===")
for q in queries:
    print(f'\n{q}:')
    print(f'  sd2vec: {[w for w, _ in sd.most_similar(q, topn=5)]}')
    print(f'  chiVe:  {[w for w, _ in w2v.most_similar(q, topn=5)]}')
    print(f'  hybrid: {[w for w, _ in hybrid.most_similar(q, topn=5)]}')

print("\n=== Diversity (100 samples, topn=10) ===")
print(f'sd2vec:  {avg_diversity(sd, sample_words):.4f}')
print(f'chiVe:   {avg_diversity(w2v, sample_words):.4f}')
print(f'hybrid:  {avg_diversity(hybrid, sample_words):.4f}')
