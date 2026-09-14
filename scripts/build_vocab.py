# -*- coding: utf-8 -*-
"""
字符级分词 + 训练/验证集切分
用法: python build_vocab.py [语料文件名] [输出前缀]
示例: python build_vocab.py wuyan_corpus.txt wuyan_   # 生成 wuyan_vocab.json 等
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # 写诗模型根目录（脚本在 scripts/ 下）


def data_dir(prefix: str) -> Path:
    """按数据前缀映射到对应子目录"""
    mapping = {
        "": ROOT / "通用模型",
        "wuyan_": ROOT / "归档\五言数据",
        "jueju_": ROOT / "绝句模型",
    }
    return mapping.get(prefix, ROOT)


BLOCK_SIZE = 128      # 上下文长度（字符数）
VAL_FRACTION = 0.02   # 验证集比例（按诗块抽样）
RANDOM_SEED = 42


def main():
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else "poetry_corpus.txt"
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    CORPUS = ROOT / "语料" / corpus_name
    VOCAB = data_dir(prefix) / f"{prefix}vocab.json"
    META = data_dir(prefix) / f"{prefix}meta.json"

    text = CORPUS.read_text(encoding="utf-8")
    print(f"语料: {corpus_name} | 总字符数: {len(text):,}")

    # 建字符词表
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size}")

    VOCAB.write_text(json.dumps(itos, ensure_ascii=False, indent=1), encoding="utf-8")

    # 按诗块（空行分隔）切分，保证一首诗不会被切开
    blocks = [b for b in text.split("\n\n") if b.strip()]
    rng = np.random.default_rng(RANDOM_SEED)
    n_val = max(1, int(len(blocks) * VAL_FRACTION))
    idx = rng.permutation(len(blocks))
    val_blocks = [blocks[i] for i in idx[:n_val]]
    train_blocks = [blocks[i] for i in idx[n_val:]]

    def encode(bs):
        return [stoi[c] for c in "\n\n".join(bs)]

    train_ids = encode(train_blocks)
    val_ids = encode(val_blocks)
    print(f"训练块: {len(train_blocks)} | 验证块: {len(val_blocks)}")
    print(f"训练 token 数: {len(train_ids):,} | 验证 token 数: {len(val_ids):,}")

    # 存为 uint16（词表 < 65536 即可）
    np.array(train_ids, dtype=np.uint16).tofile(data_dir(prefix) / f"{prefix}train.bin")
    np.array(val_ids, dtype=np.uint16).tofile(data_dir(prefix) / f"{prefix}val.bin")
    META.write_text(json.dumps({
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成: {prefix}vocab.json / {prefix}train.bin / {prefix}val.bin / {prefix}meta.json")


if __name__ == "__main__":
    main()
