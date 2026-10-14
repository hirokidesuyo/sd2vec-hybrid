"""GitHub Pages 用のデータを書き出す。

SD ベクトルと chiVe ベクトルの共通語彙を、L2 正規化したうえで float16 バイナリとして
docs/ に出力する。ハイブリッド(正規化済みベクトルの連結)のコサイン類似度は
(SD内積 + chiVe内積) / 2 と等価なので、ブラウザ側で計算し、ここでは保存しない。

出力:
  words.json   語彙リスト
  sd.f16       (N, sd_dim)  float16 行優先
  w2v.f16      (N, w2v_dim) float16 行優先
  meta.json    {"count", "sd_dim", "w2v_dim"}
"""
import argparse
import json
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from sd2vec import SDVectors


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def main():
    parser = argparse.ArgumentParser(description="Export web data for GitHub Pages")
    parser.add_argument("--sd", required=True, help="Path to sd2vec model directory")
    parser.add_argument("--w2v", required=True, help="Path to chiVe KeyedVectors (.kv)")
    parser.add_argument("--output", default="docs", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max number of words")
    args = parser.parse_args()

    sd = SDVectors.load(args.sd)
    w2v = KeyedVectors.load(args.w2v, mmap="r")

    # SD 側の語順(頻度順)を保ち、chiVe にも存在する語だけを残す
    words = [w for w in sd.index_to_key if w in w2v.key_to_index]
    if args.limit:
        words = words[: args.limit]

    sd_mat = normalize_rows(np.stack([np.asarray(sd.get_vector(w)) for w in words]).astype(np.float32))
    w2v_mat = normalize_rows(np.stack([np.asarray(w2v[w]) for w in words]).astype(np.float32))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "words.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    sd_mat.astype("<f2").tofile(out / "sd.f16")
    w2v_mat.astype("<f2").tofile(out / "w2v.f16")
    (out / "meta.json").write_text(
        json.dumps({"count": len(words), "sd_dim": sd_mat.shape[1], "w2v_dim": w2v_mat.shape[1]}),
        encoding="utf-8",
    )
    print(f"Exported {len(words)} words to {out}")


if __name__ == "__main__":
    main()
