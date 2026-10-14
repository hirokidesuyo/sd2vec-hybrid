"""GitHub Pages 用のデータを書き出す。

SD ベクトルと chiVe ベクトルの共通語彙を float16 バイナリとして docs/ に出力する。
SD は各軸の得点(-3〜3)をそのまま保存し、chiVe は L2 正規化して保存する。
ハイブリッド(正規化済みベクトルの連結)のコサイン類似度は
(SD内積 + chiVe内積) / 2 と等価なので、SD の正規化とあわせてブラウザ側で計算する。

出力:
  words.json   語彙リスト
  axes.json    SD 軸の定義 [{"id", "left", "right"}]。得点が正なら left、負なら right
  sd.f16       (N, sd_dim)  float16 行優先(正規化前の得点)
  w2v.f16      (N, w2v_dim) float16 行優先(L2 正規化済み)
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

    # SD は各軸の意味(-3〜3)を表示に使うため、正規化せず生の得点のまま保存する
    sd_mat = np.stack([np.asarray(sd.get_vector(w)) for w in words]).astype(np.float32)
    w2v_mat = normalize_rows(np.stack([np.asarray(w2v[w]) for w in words]).astype(np.float32))

    # 軸の定義。得点が正なら left、負なら right の印象に近い
    metadata = json.loads((Path(args.sd) / "metadata.json").read_text(encoding="utf-8"))
    axes = [{"id": d["id"], "left": d["left"], "right": d["right"]} for d in metadata["dimensions"]]
    assert len(axes) == sd_mat.shape[1], "axes and SD vector dimension mismatch"

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "words.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    (out / "axes.json").write_text(json.dumps(axes, ensure_ascii=False), encoding="utf-8")
    sd_mat.astype("<f2").tofile(out / "sd.f16")
    w2v_mat.astype("<f2").tofile(out / "w2v.f16")
    (out / "meta.json").write_text(
        json.dumps({"count": len(words), "sd_dim": sd_mat.shape[1], "w2v_dim": w2v_mat.shape[1]}),
        encoding="utf-8",
    )
    print(f"Exported {len(words)} words to {out}")


if __name__ == "__main__":
    main()
