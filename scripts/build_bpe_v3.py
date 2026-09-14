# -*- coding: utf-8 -*-
"""
v3 词表构建：精炼纯字表（Top 4000 高频古诗字符 + 5 个特殊 token）
============================================================
与 v2 的 BPE 子词方案相反：中国古典诗歌"一字一音一拍"，五言诗严格 5 token。
BPE 把"春风""孤舟"合并成 1 个 token 后模型无法显式数数 → 句长飘忽。
v3 回归字符级，但词表从全量 1 万+ 瘦身到 4000 高频字符（覆盖率 99.85%），
省出的 90 万嵌入参数让给更深更宽的 Transformer 骨架。

数据格式（特殊 token 控制）:
  <|bos|>《标题》<|title|>作者<|author|><|body|>诗句（含句读标点）<|eos|>

用法: python scripts\build_bpe_v3.py
输出: 五言BPE_v3\\wuyan_bpe_v3_vocab.json / train.bin / val.bin / meta.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "语料"
OUT_DIR = ROOT / "五言BPE_v3"
PREFIX = "wuyan_bpe_v3_"

BLOCK_SIZE = 128
VAL_FRACTION = 0.02
RANDOM_SEED = 42
TOP_CHARS = 4000          # 高频字符（含汉字与句读标点《》，。等）
# 6 个特殊 token：bos/title/author/body/eos + unk 兜底（表外生僻字→unk，不丢数据）
SPECIAL_TOKENS = ["<|bos|>", "<|title|>", "<|author|>", "<|body|>", "<|eos|>", "<|unk|>"]

# 特殊 token id: 0..5，字符从 6 开始
BOS, TITLE, AUTHOR, BODY, EOS, UNK = range(6)

_PUNCT = "，。！？；、"
_POLLUTE = ("诗题", "首句", "全诗）", "全诗)", "（全诗）", "已佚", "……", "残句")


def clean_block(block: str):
    """解析一个诗块 → (title, author, [原始诗句行])；不合格返回 None"""
    raw_lines = [l.strip() for l in block.split("\n") if l.strip()]
    title, author = None, None
    verse_lines = []
    for ln in raw_lines:
        if any(p in ln for p in _POLLUTE):
            continue
        if "《" in ln or "》" in ln:
            title = ln
        elif title is not None and author is None and len(ln) <= 4 and not any(p in ln for p in _PUNCT):
            author = ln
        else:
            # 严格五言：该行按句读切出的每个片段都必须正好 5 字
            segs = [s.strip() for s in re.split(r"[，。！？；、]", ln.replace(" ", "")) if s.strip()]
            if segs and all(len(s) == 5 for s in segs):
                verse_lines.append(ln.replace(" ", ""))   # 保留原始句读（逗号/句号分布）
    if title is None or author is None or len(verse_lines) < 4:
        return None
    return title, author, verse_lines


def main():
    text = (CORPUS_DIR / "wuyan_corpus.txt").read_text(encoding="utf-8")
    blocks = [b for b in text.split("\n\n") if b.strip()]
    print(f"原始诗块: {len(blocks):,}")

    # 1) 清洗 + 严格五言过滤
    poems = []
    for b in blocks:
        p = clean_block(b)
        if p:
            poems.append(p)
    print(f"清洗后(标题+作者+≥4个五言句): {len(poems):,} 首")

    # 2) 去重（小模型经验：数据质量 > 数据量）
    seen, dedup = set(), []
    for p in poems:
        h = hash("|".join(p[0]) + "|" + p[1] + "|" + "|".join(p[2]))
        if h not in seen:
            seen.add(h)
            dedup.append(p)
    print(f"去重后: {len(dedup):,} 首（去掉了 {len(poems) - len(dedup):,} 首重复）")

    # 3) 字符频次统计 → Top 4000（汉字 + 句读标点都参与排序，超高频标点自然入选）
    cnt = Counter()
    for title, author, segs in dedup:
        cnt.update(title)
        cnt.update(author)
        for s in segs:
            cnt.update(s)
    top = [c for c, _ in cnt.most_common(TOP_CHARS)]
    top_set = set(top)
    print(f"高频字符表: {len(top)}（覆盖 {sum(cnt[c] for c in top) / sum(cnt.values()):.4%} 语料字符）")
    print("  前 20 高频:", "".join(top[:20]))

    # 4) 词表：5 特殊 token + 4000 字符
    itos = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
    for i, c in enumerate(top, start=len(SPECIAL_TOKENS)):
        itos[i] = c
    stoi = {t: i for i, t in itos.items()}
    vocab_size = len(itos)
    print(f"词表大小: {vocab_size}（{TOP_CHARS} 字符 + {len(SPECIAL_TOKENS)} 特殊 token）")

    # 5) 编码（表外字 → <|unk|> 兜底，数据全量保留）
    def encode_poem(p):
        title, author, lines = p
        def enc(s):
            return [stoi.get(c, UNK) for c in s]
        # 保留原始诗句行与句读，最多 8 行（约 100 字符，保证不超 block 128）
        body = "".join(lines[:8])[:100]
        return ([BOS] + enc(title) + [TITLE]
                + enc(author) + [AUTHOR, BODY]
                + enc(body) + [EOS])

    encoded = [encode_poem(p) for p in dedup]
    print(f"编码完成: {len(encoded):,} 首（表外字已映射到 <|unk|>，数据无丢失）")

    # 6) 切分 train/val 并落盘
    rng = np.random.default_rng(RANDOM_SEED)
    n_val = max(1, int(len(encoded) * VAL_FRACTION))
    idx = rng.permutation(len(encoded))
    val_blocks = [encoded[i] for i in idx[:n_val]]
    train_blocks = [encoded[i] for i in idx[n_val:]]

    def flatten(bs):
        out = []
        for e in bs:
            out.extend(e)
        return np.array(out, dtype=np.uint16)

    train_ids, val_ids = flatten(train_blocks), flatten(val_blocks)
    print(f"训练 token: {len(train_ids):,} | 验证 token: {len(val_ids):,} | 平均每首 {len(train_ids)/len(train_blocks):.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{PREFIX}vocab.json").write_text(json.dumps(itos, ensure_ascii=False), encoding="utf-8")
    train_ids.tofile(OUT_DIR / f"{PREFIX}train.bin")
    val_ids.tofile(OUT_DIR / f"{PREFIX}val.bin")
    (OUT_DIR / f"{PREFIX}meta.json").write_text(json.dumps({
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "num_chars": TOP_CHARS,
        "special_tokens": SPECIAL_TOKENS,
        "dedup": len(encoded),
        "tokenizer": "char_top4000_special",
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # 样例预览
    print("\n样例编码:")
    for i, e in enumerate(encoded[:2]):
        print("  ", "".join(itos[t] for t in e))
    print(f"\n完成: {OUT_DIR / (PREFIX + 'vocab.json')} / train.bin / val.bin / meta.json")


if __name__ == "__main__":
    main()
