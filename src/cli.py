import argparse
from .hybrid_vectors import HybridVectors


def main():
    parser = argparse.ArgumentParser(description="sd2vec-hybrid CLI")
    sub = parser.add_subparsers(dest="command")

    neighbors = sub.add_parser("neighbors", help="Find nearest neighbors")
    neighbors.add_argument("sd_path", help="Path to sd2vec model")
    neighbors.add_argument("w2v_path", help="Path to word2vec model")
    neighbors.add_argument("word", help="Query word")
    neighbors.add_argument("--topn", type=int, default=10)

    args = parser.parse_args()

    if args.command == "neighbors":
        model = HybridVectors.load(args.sd_path, args.w2v_path)
        print(f"vector_size={model.vector_size}")
        for word, score in model.most_similar(args.word, topn=args.topn):
            print(f"{word}\t{score:.6f}")


if __name__ == "__main__":
    main()
