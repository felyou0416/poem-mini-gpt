# -*- coding: utf-8 -*-
"""
整理 chinese-poetry 语料为模型训练文本
输入: chinese-poetry 各分类目录的 JSON
输出: poetry_corpus.txt (统一格式: 《标题》/作者/诗句.../空行)
"""
import json
from pathlib import Path

try:
    from opencc import OpenCC
    cc = OpenCC('t2s')
    HAS_OPENCC = True
    print("[i] 简体转换: 已启用 (t2s)")
except Exception as e:
    HAS_OPENCC = False
    print("[!] 简体转换不可用:", e)

ROOT = Path(__file__).resolve().parent.parent   # 写诗模型根目录
BASE = ROOT / "chinese-poetry"
OUT_DIR = ROOT / "语料"
OUT_TXT = OUT_DIR / "poetry_corpus.txt"

# 选取的语料目录（四书五经/论语/蒙学等非诗歌类不选）
DIRS = ["全唐诗", "宋词", "元曲", "诗经", "楚辞", "五代诗词", "曹操诗集", "纳兰性德"]

MIN_PARAS = 2      # 最少诗句数
MIN_CHARS = 20     # 最少总字数
MAX_TITLE = 30     # 标题最长字符数（过滤文献题记类超长标题）


def to_simplified(s: str) -> str:
    return cc.convert(s) if HAS_OPENCC else s


def parse_json_file(path: Path):
    """解析单个 JSON 文件，返回诗歌列表"""
    poems = []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return poems

    # 兼容 {poems:[...]} 之类的字典包裹
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                data = v
                break
    if not isinstance(data, list):
        return poems

    for it in data:
        if not isinstance(it, dict):
            continue
        paras = it.get("paragraphs") or it.get("content") or it.get("para")
        if not isinstance(paras, list) or len(paras) < MIN_PARAS:
            continue
        paras = [str(p).strip() for p in paras if str(p).strip()]
        if len(paras) < MIN_PARAS:
            continue
        title = str(it.get("title") or it.get("rhythmic") or "").strip()
        author = str(it.get("author") or "").strip()
        # 过滤噪声：标题过长（文献题记）、标题或诗句含缺字符□
        if len(title) > MAX_TITLE:
            continue
        if any("\ufffd" in x or "□" in x for x in paras + [title, author]):
            continue
        poems.append({"title": title, "author": author, "paras": paras})
    return poems


def fmt_poem(p) -> str:
    lines = []
    if p["title"]:
        lines.append(f"《{p['title']}》")
    if p["author"]:
        lines.append(p["author"])
    lines.extend(p["paras"])
    return "\n".join(lines) + "\n\n"


def main():
    total_poems = 0
    total_chars = 0
    stat = {}
    with open(OUT_TXT, "w", encoding="utf-8") as out:
        for d in DIRS:
            dpath = BASE / d
            if not dpath.exists():
                print(f"[!] 目录不存在: {d}")
                continue
            json_files = sorted(dpath.rglob("*.json"))
            cnt = 0
            for jf in json_files:
                for p in parse_json_file(jf):
                    body = "".join(p["paras"])
                    if len(body) < MIN_CHARS:
                        continue
                    text = fmt_poem({
                        "title": to_simplified(p["title"]),
                        "author": to_simplified(p["author"]),
                        "paras": [to_simplified(x) for x in p["paras"]],
                    })
                    out.write(text)
                    cnt += 1
                    total_chars += len(text)
            total_poems += cnt
            stat[d] = cnt
            print(f"  {d}: {cnt} 首")

    print("=" * 40)
    print(f"总诗数: {total_poems}")
    print(f"总字符数: {total_chars:,}")
    print(f"输出: {OUT_TXT} ({OUT_TXT.stat().st_size/1024/1024:.1f} MB)")
    with open(OUT_DIR / "corpus_stat.json", "w", encoding="utf-8") as f:
        json.dump({"total_poems": total_poems, "total_chars": total_chars, "by_dir": stat}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
