# -*- coding: utf-8 -*-
"""
BPE 子词分词器（从零实现，教学版）
====================================
原理：从字符开始，反复合并"出现频率最高"的相邻字符对，学到 N 次合并后，
     就把常用的词/词组（如山、水流、春风）变成了单个 token。

三步：
  1. 训练：在语料采样上学出 merges 合并规则
  2. 去重：按诗内容哈希去掉重复诗（小模型经验：数据质量 > 数据量）
  3. 编码：把全量语料切成子词 token → train.bin / val.bin / vocab.json / meta.json

用法: python scripts\build_bpe.py <语料文件名> <输出前缀> [合并次数] [训练采样字符数]
示例: python scripts\build_bpe.py wuyan_corpus.txt wuyan_bpe_ 2000 1000000

输出（到 五言BPE/ 目录，与 build_vocab.py 格式兼容）:
  {前缀}vocab.json   {id: token字符串}，字符 0..C-1，合并token 从 C 开始
  {前缀}train.bin / val.bin / meta.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "语料"

BLOCK_SIZE = 128
VAL_FRACTION = 0.02
RANDOM_SEED = 42
KEY_SHIFT = 20  # 相邻 pair 打包成单个 int 键的位数（词表远小于 2^20）


def data_dir(prefix: str) -> Path:
    mapping = {
        "": ROOT / "通用模型",
        "wuyan_": ROOT / "归档\五言数据",
        "wuyan_bpe_": ROOT / "五言BPE",
        "jueju_": ROOT / "绝句模型",
    }
    return mapping.get(prefix, ROOT)


# ---------- 1. BPE 训练 ----------
def train_bpe(text: str, num_merges: int, sample_chars: int, chars: list):
    """学 merges：反复合并最高频相邻字符对
    chars: 全量语料的完整字符表（保证编码时生僻字也有 id）
    """
    if sample_chars and len(text) > sample_chars:
        text = text[:sample_chars]
        print(f"  BPE 训练采样: {sample_chars:,} 字符")
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    C = len(chars)
    merges = []

    for m in range(num_merges):
        # 统计相邻 pair 频次（打包成 int 键，numpy 加速）
        pairs = (ids[:-1].astype(np.int64) << KEY_SHIFT) | ids[1:]
        uniq, counts = np.unique(pairs, return_counts=True)
        if len(uniq) == 0:
            break
        best_key = int(uniq[int(np.argmax(counts))])
        a, b = best_key >> KEY_SHIFT, best_key & ((1 << KEY_SHIFT) - 1)
        # 合并所有出现处（a,b → 新 token id）
        new_token = C + m
        new_ids = []
        app = new_ids.append
        i, n = 0, len(ids)
        while i < n:
            if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                app(new_token)
                i += 2
            else:
                app(int(ids[i]))
                i += 1
        ids = np.array(new_ids, dtype=np.int32)
        merges.append((int(a), int(b)))
        if (m + 1) % 200 == 0:
            print(f"  merge {m + 1}/{num_merges} | 序列长度 {len(ids):,}")
    return chars, merges


# ---------- 2. 编码（单个块） ----------
def encode_block(ids: list, merges: list, rank: dict):
    """对单块 id 序列，按 merge 顺序（rank 越小越先）贪心合并"""
    while True:
        best_i, best_r = None, None
        for i in range(len(ids) - 1):
            r = rank.get((ids[i], ids[i + 1]))
            if r is not None and (best_r is None or r < best_r):
                best_i, best_r = i, r
        if best_i is None:
            break
        ids = ids[:best_i] + [rank[(ids[best_i], ids[best_i + 1])]] + ids[best_i + 2:]
    return ids


# ---------- 3. 主流程 ----------
def main():
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else "wuyan_corpus.txt"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "wuyan_bpe_"
    num_merges = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    sample_chars = int(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000

    text = (CORPUS_DIR / corpus_name).read_text(encoding="utf-8")
    print(f"语料: {corpus_name} | 总字符数: {len(text):,}")

    # 按诗块切分
    blocks = [b for b in text.split("\n\n") if b.strip()]
    print(f"诗块总数: {len(blocks):,}")

    # 去重（小模型经验：数据质量优先；不同诗集可能有重复诗）
    seen = set()
    dedup = []
    for b in blocks:
        h = hash(b)
        if h not in seen:
            seen.add(h)
            dedup.append(b)
    print(f"去重后: {len(dedup):,} 首（去掉了 {len(blocks) - len(dedup):,} 首重复）")

    # BPE 训练（用去重后的文本；字符表覆盖全量，保证生僻字可编码）
    train_text = "\n\n".join(dedup)
    full_chars = sorted(set(train_text))
    print(f"全量字符表: {len(full_chars):,} 字符")
    print("开始 BPE 训练...")
    chars, merges = train_bpe(train_text, num_merges, sample_chars, full_chars)
    C = len(chars)
    print(f"BPE 训练完成: 字符 {C} + 合并 {len(merges)} = 词表 {C + len(merges):,}")

    # 构建词表与 rank
    itos = {i: c for i, c in enumerate(chars)}
    for m, (a, b) in enumerate(merges):
        itos[C + m] = itos[a] + itos[b]  # 合并 token 的字符串 = 两段拼接
    stoi = {tok: i for i, tok in itos.items()}
    rank = {pair: C + m for m, pair in enumerate(merges)}

    # 编码（按诗块编码，块间补 \n\n）
    rng = np.random.default_rng(RANDOM_SEED)
    n_val = max(1, int(len(dedup) * VAL_FRACTION))
    idx = rng.permutation(len(dedup))
    val_blocks = [dedup[i] for i in idx[:n_val]]
    train_blocks = [dedup[i] for i in idx[n_val:]]

    def encode_many(bs):
        out = []
        nl = stoi["\n"]
        for b in bs:
            out.extend(encode_block([stoi[c] for c in b], merges, rank))
            out.extend([nl, nl])
        return out

    print("编码训练集...")
    train_ids = encode_many(train_blocks)
    print(f"训练 token: {len(train_ids):,}")
    print("编码验证集...")
    val_ids = encode_many(val_blocks)
    print(f"验证 token: {len(val_ids):,}")

    out_dir = data_dir(prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}vocab.json").write_text(json.dumps(itos, ensure_ascii=False), encoding="utf-8")
    np.array(train_ids, dtype=np.uint16).tofile(out_dir / f"{prefix}train.bin")
    np.array(val_ids, dtype=np.uint16).tofile(out_dir / f"{prefix}val.bin")
    (out_dir / f"{prefix}meta.json").write_text(json.dumps({
        "vocab_size": C + len(merges),
        "block_size": BLOCK_SIZE,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "num_merges": len(merges),
        "num_chars": C,
        "dedup": len(dedup),
        "tokenizer": "bpe",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成: {prefix}vocab.json / train.bin / val.bin / meta.json（{prefix} 共 {C + len(merges):,} tokens）")


if __name__ == "__main__":
    main()
