# sd2vec-hybrid

**sd2vec-hybrid** は、sd2vec（意味微分法ベースの解釈可能な概念ベクトル）とword2vec（chiVeなどの分布意味ベクトル）を正規化して連結し、相補的なベクトル表現を作成するツールです。

## 特徴

- sd2vec（50次元・SD尺度）とword2vec（300次元・文脈的意味）を正規化して連結
- 各ベクトルの次元数とスケールの違いを吸収
- 結合空間での最近傍検索をサポート
- シンプルなPython APIとCLI

## インストール

```bash
git clone https://github.com/hirokidesuyo/sd2vec-hybrid.git
cd sd2vec-hybrid
python -m pip install -e .
```

## 使い方

### Python API

```python
from sd2vec_hybrid import HybridVectors

model = HybridVectors.load(
    sd_path="path/to/outputs_chive_10000_phase3",
    w2v_path="path/to/chive_vectors"
)

print(model.vector_size)            # 350 (50 + 300)
print(model.index_to_key[:3])       # 語彙の先頭
print(model.get_vector("猫"))       # numpy.ndarray(shape=(350,))
print(model.most_similar("猫", topn=5))
```

### CLI

```bash
sd2vec-hybrid-neighbors path/to/sd2vec path/to/word2vec 猫 --topn 10
```

## 出力形式

結合ベクトルは `numpy` 形式とUTF-8 CSV形式で保存できます。

```python
model.save_npy("hybrid_vectors.npy")
model.save_csv("hybrid_vectors.csv")
```

## ライセンス

MIT License
