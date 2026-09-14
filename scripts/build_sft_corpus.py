# -*- coding: utf-8 -*-
"""
P2 名篇 SFT 语料构建：从全唐诗提取名家五言传世篇目
================================================
数据源: chinese-poetry/全唐诗/poet.tang.*.json（繁体）
目标: 李白/杜甫/王维/孟浩然/韦应物/刘长卿/李商隐/王昌龄/白居易/岑参/韩愈/
     柳宗元/刘禹锡/杜牧/贾岛/孟郊/张九龄/初唐四杰/陈子昂/宋之问/沈佺期等
筛选: 简体转换后每句恰 5 字（严格五言）、4~8 句（绝句/律诗/短古）
输出: 语料/sft_tang_corpus.txt（与预训练同格式：《题》/作者/诗句行）
用途: train_sft.py 以 v3 预训练 checkpoint 为起点，LR 5e-5 轻量微调
"""
import json
import re
import sys
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parent.parent
TANG_DIR = ROOT / "chinese-poetry" / "全唐诗"
OUT = ROOT / "语料" / "sft_tang_corpus.txt"

TARGETS = {
    "李白", "杜甫", "王维", "孟浩然", "韦应物", "刘长卿", "李商隐", "王昌龄",
    "白居易", "岑参", "韩愈", "柳宗元", "刘禹锡", "杜牧", "贾岛", "孟郊",
    "张九龄", "王勃", "杨炯", "卢照邻", "骆宾王", "陈子昂", "宋之问",
    "沈佺期", "储光羲", "常建", "张籍", "元稹", "许浑", "温庭筠",
}
# 作者配额：防止白居易等巨量诗作压扁风格多样性（名篇=质量均衡）
QUOTA = {"李白": 600, "杜甫": 600, "白居易": 500}
DEFAULT_QUOTA = 400


def main():
    cc = OpenCC("t2s")
    out_lines = []
    stats = {}
    n_total = n_picked = 0

    for f in sorted(TANG_DIR.glob("poet.tang.*.json")):
        for p in json.load(open(f, encoding="utf-8")):
            author = cc.convert(p.get("author", "")).strip()
            if author not in TARGETS:
                continue
            n_total += 1
            stats[author] = stats.get(author, 0) + 1

            # 配额控制：超出该作者上限则跳过（保持各家风格均衡）
            if stats[author] > QUOTA.get(author, DEFAULT_QUOTA):
                continue

            lines = [cc.convert(l).replace(" ", "") for l in p.get("paragraphs", [])]
            # 严格五言：每行按句读切分后每个片段恰 5 字
            segs = [s for ln in lines for s in re.split(r"[，。！？；、]", ln) if s.strip()]
            if not segs or any(len(s) != 5 for s in segs):
                continue
            if not (4 <= len(segs) <= 8):
                continue
            title = p.get("title", "").strip()
            if not title or any(ch in title for ch in "诗题首句"):
                continue
            n_picked += 1
            out_lines.append(f"《{title}》")
            out_lines.append(author)
            out_lines.extend(lines)
            out_lines.append("")

    OUT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"输出: {OUT}")
    print(f"目标作者诗作总数: {n_total:,} | 严格五言 4-8 句名篇: {n_picked:,}")
    print("作者分布 Top10:", sorted(stats.items(), key=lambda x: -x[1])[:10])
    print("样例:")
    sample = [l for l in out_lines if l][:6]
    print("  " + "\n  ".join(sample))


if __name__ == "__main__":
    main()
