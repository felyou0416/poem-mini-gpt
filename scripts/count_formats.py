# -*- coding: utf-8 -*-
"""统计语料中各格式诗歌数量与占比（按分句字数归类）"""
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "语料" / "poetry_corpus.txt").read_text(encoding="utf-8")
blocks = [b for b in text.split("\n\n") if b.strip()]

PUNCT = set("，。；、？！：·—-～…“”\"'『』《》（）【】")
CLAUSE_SEP = re.compile(r"[，。；、？！]")


def clean(s: str) -> str:
    return "".join(c for c in s if c not in PUNCT)


stat = Counter()
for b in blocks:
    lines = [x.strip() for x in b.split("\n") if x.strip()]
    if len(lines) < 2:
        continue
    # lines[0] 是标题《...》；其余行中，作者行无标点且短，诗句行有标点或较长
    poem_lines = []
    for ln in lines[1:]:
        if "《" in ln or "》" in ln:
            continue
        if not any(p in ln for p in "，。；、？！") and len(ln) <= 6:
            continue  # 作者行
        poem_lines.append(ln)
    if not poem_lines:
        continue

    clauses = []
    for ln in poem_lines:
        for part in CLAUSE_SEP.split(ln):
            c = clean(part)
            if c:
                clauses.append(c)
    if not clauses:
        continue

    lens = [len(x) for x in clauses]
    main = Counter(lens).most_common(1)[0][0]
    ratio = Counter(lens)[main] / len(lens)
    if ratio >= 0.75 and main in (5, 7, 6, 4):
        label = {5: "五言", 7: "七言", 6: "六言", 4: "四言"}[main]
    else:
        label = "杂言(词/曲/长短句)"
    stat[label] += 1

total = sum(stat.values())
print(f"总诗数: {total:,}")
for k, v in stat.most_common():
    print(f"{k}: {v:,} 首 ({v / total * 100:.1f}%)")
