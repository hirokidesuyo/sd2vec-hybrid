import argparse
from sd2vec_hybrid import HybridVectors


def main():
    parser = argparse.ArgumentParser(description="Build hybrid vectors from sd2vec and word2vec")
    parser.add_argument("--sd", required=True, help="Path to sd2vec model")
    parser.add_argument("--w2v", required=True, help="Path to word2vec model")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    model = HybridVectors.load(args.sd, args.w2v)

    import os
    os.makedirs(args.output, exist_ok=True)
    model.save_npy(os.path.join(args.output, "hybrid_vectors.npy"))
    model.save_csv(os.path.join(args.output, "hybrid_vectors.csv"))

    print(f"Saved {len(model.index_to_key)} vectors ({model.vector_size} dims) to {args.output}")


if __name__ == "__main__":
    main()
