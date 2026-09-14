# -*- coding: utf-8 -*-
"""
从 poetry_corpus.txt 筛出五言诗，生成 wuyan_corpus.txt
判定：诗句分句长度主体为 5 字（>=75%），保留原格式
用法: python filter_wuyan.py [分句数] [输出文件名]
示例: python filter_wuyan.py           # 全部五言 → wuyan_corpus.txt
      python filter_wuyan.py 4 jueju   # 五言四句(绝句) → jueju_corpus.txt
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "语料" / "poetry_corpus.txt"

PUNCT = set("，。；、？！：·—-～…“”\"'『』《》（）【】")
CLAUSE_SEP = re.compile(r"[，。；、？！]")


def clean(s: str) -> str:
    return "".join(c for c in s if c not in PUNCT)


def classify(block: str):
    """返回 (是否五言, 分句数)"""
    lines = [x.strip() for x in block.split("\n") if x.strip()]
    if len(lines) < 2:
        return False, 0
    poem_lines = []
    for ln in lines[1:]:
        if "《" in ln or "》" in ln:
            continue
        if not any(p in ln for p in "，。；、？！") and len(ln) <= 6:
            continue  # 作者行
        poem_lines.append(ln)
    if not poem_lines:
        return False, 0
    clauses = []
    for ln in poem_lines:
        for part in CLAUSE_SEP.split(ln):
            c = clean(part)
            if c:
                clauses.append(c)
    if not clauses:
        return False, 0
    lens = [len(x) for x in clauses]
    main = Counter(lens).most_common(1)[0][0]
    ratio = Counter(lens)[main] / len(lens)
    is_wuyan = main == 5 and ratio >= 0.75
    return is_wuyan, len(clauses)


def main():
    n_clauses = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = 全部五言
    out_name = sys.argv[2] if len(sys.argv) > 2 else "wuyan"
    OUT = ROOT / "语料" / f"{out_name}_corpus.txt"

    text = SRC.read_text(encoding="utf-8")
    blocks = [b for b in text.split("\n\n") if b.strip()]
    kept = []
    for b in blocks:
        ok, n = classify(b)
        if ok and (n_clauses == 0 or n == n_clauses):
            kept.append(b)
    OUT.write_text("\n\n".join(kept) + "\n", encoding="utf-8")
    chars = sum(len(b) for b in kept)
    tag = f"五言{n_clauses}句" if n_clauses else "五言"
    print(f"{tag}诗: {len(kept):,} 首 | 约 {chars:,} 字符 | {OUT.name} ({OUT.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
