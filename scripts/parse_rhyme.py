# -*- coding: utf-8 -*-
"""从 _rhyme_raw.txt 解析中华新韵十四韵韵字，生成 rhyme_dict.py"""
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent / "_rhyme_raw.txt"
OUT = Path(__file__).resolve().parent / "rhyme_dict.py"

RHYME_ORDER = ["一麻", "二波", "三皆", "四开", "五微", "六豪", "七尤",
               "八寒", "九文", "十唐", "十一庚", "十二齐", "十三支", "十四姑"]
SHORT2FULL = {"麻": "一麻", "波": "二波", "皆": "三皆", "开": "四开", "微": "五微",
              "豪": "六豪", "尤": "七尤", "寒": "八寒", "文": "九文", "唐": "十唐",
              "庚": "十一庚", "齐": "十二齐", "支": "十三支", "姑": "十四姑"}


def main():
    text = RAW.read_text(encoding="utf-8")
    # 提取 <link ... text="xxx" /> 中的文字，替换回原文
    text = re.sub(r'<link[^>]*text="([^"]*)"[^>]*/>', r"\1", text)
    # 去掉残余标签
    text = re.sub(r"<[^>]+>", "", text)

    # 定位各韵部正文（标题后紧跟【或（）
    positions = []
    for t in RHYME_ORDER:
        # 找到标题后带【的位置（正文），跳过顶部导航
        for m in re.finditer(re.escape(t), text):
            after = text[m.end():m.end() + 10]
            if "【" in after:  # 正文标题后紧跟全角【；顶部导航无【，跳过
                positions.append((m.start(), t))
                break
    positions.sort()
    if len(positions) != 14:
        print(f"!! 只找到 {len(positions)} 个韵部:", [t for _, t in positions])
        return

    rhymes = {t: [] for t in RHYME_ORDER}
    for i, (start, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        seg = text[start:end]
        # 从【】之后开始（跳过韵母说明）
        j = seg.find("】")
        seg = seg[j + 1:] if j >= 0 else seg

        # 1) 去掉括号说明后，按声调标签切分：丢弃标签本身，保留标签后的韵字
        seg_clean = re.sub(r"[（(][^）)]*[)）]", "", seg)
        seg_clean = seg_clean.replace("零韵母", "")  # 十三支【－i】后的说明文字
        seg_clean = seg_clean.replace("发动", "发")  # 原网页排版错字（十二齐入声字）
        seg_clean = seg_clean.replace("放过", "")   # 原网页排版错字（十一庚阳平"廷亭放过停蜓"）
        parts = re.split(r"(派入[^：]*?入声字：|阴平：|阳平：|上声：|去声：)", seg_clean)
        content = "".join(p for i, p in enumerate(parts) if i % 2 == 0)
        chars = re.findall(r"[\u4e00-\u9fff]", content)
        # 先归本韵（本韵字在前），多音并入字在后
        for ch in chars:
            if ch not in rhymes[title]:
                rhymes[title].append(ch)
        # 2) 多音字并入（放本韵字之后）
        for m in re.finditer(r"[（(]又([^）)]*?韵)[^）)]*[)）]", seg):
            prefix = seg[:m.start()]
            prev_chars = re.findall(r"[\u4e00-\u9fff]", prefix)
            if not prev_chars:
                continue
            ch = prev_chars[-1]
            for name in re.findall(r"[\u4e00-\u9fff]+", m.group(1)):
                name = name.replace("韵", "")
                for part in re.split(r"[、,，]", name):
                    full = SHORT2FULL.get(part)
                    if full and ch not in rhymes[full]:
                        rhymes[full].append(ch)
        for ch in chars:
            if ch not in rhymes[title]:
                rhymes[title].append(ch)

    total = sum(len(v) for v in rhymes.values())
    for t in RHYME_ORDER:
        print(f"{t}: {len(rhymes[t])} 字")
    print(f"总计: {total} 字")

    # 生成 rhyme_dict.py
    lines = [
        "# -*- coding: utf-8 -*-",
        '"""中华新韵（十四韵）韵字表 —— P0 押韵评分器数据',
        "",
        "数据来源: 中华诗词学会《中华新韵》(2005版) 十四韵韵字表",
        "         (华韵诗词学习网整理版: http://www.huayunonline.cn/hy-list-zhxy.htm)",
        "",
        "说明:",
        "  1. 每韵部含阴平/阳平/上声/去声及派入的入声字，全部按韵部归类;",
        "  2. 多音字依“(又X韵)”标注同时归入对应韵部，提高押韵判定精度;",
        "  3. 本表只用于判定两字是否同韵部（押韵），不细分平仄。",
        "",
        "用法:",
        "    from rhyme_dict import rhyme_of, is_rhyme",
        "    rhyme_of('秋')   -> ['七尤']",
        '    is_rhyme("愁", "楼") -> True',
        '"""',
        "",
        "# 韵部 -> 韵字列表",
        "RHYME_CHARS = {",
    ]
    for t in RHYME_ORDER:
        chars_str = "".join(rhymes[t])
        lines.append(f'    "{t}": "{chars_str}",')
    lines.append("}")
    lines += [
        "",
        "# 反查: 字 -> 所属韵部列表（多音字可能跨韵部）",
        "CHAR_RHYMES = {}",
        "for _r, _cs in RHYME_CHARS.items():",
        "    for _c in _cs:",
        "        CHAR_RHYMES.setdefault(_c, []).append(_r)",
        "",
        "RHYME_ORDER = [\"一麻\", \"二波\", \"三皆\", \"四开\", \"五微\", \"六豪\", \"七尤\",",
        '               "八寒", "九文", "十唐", "十一庚", "十二齐", "十三支", "十四姑"]',
        "",
        "",
        "def rhyme_of(ch):",
        '    """返回某字的韵部列表；词表外/生僻字返回空列表"""',
        "    return CHAR_RHYMES.get(ch, [])",
        "",
        "",
        "def is_rhyme(ch1, ch2):",
        '    """两字是否可押韵（所属韵部有交集）"""',
        "    r1 = CHAR_RHYMES.get(ch1, [])",
        "    r2 = CHAR_RHYMES.get(ch2, [])",
        "    return bool(set(r1) & set(r2))",
        "",
        "",
        "# ============ P0 押韵评分器（Best-of-N 候选重排用） ============",
        "# 规则（按五言BPE_v3优化落地方案.md）:",
        "#   base 100",
        "#   - 严格五言: 每句必须正好 5 个汉字，多/少 1 字 -15",
        "#   - 二四句押韵: 第 2、4 句尾字同韵部 +40；不同韵 -40",
        "#   - 首句入韵（第 1 句尾字与韵脚同韵部）: +10",
        "#   - 句内重字: 每处 -20",
        "#   - 泛滥字（单字全诗 >3 次）: 每超 1 次 -10",
        "#   - 有效诗句不足 4 句: 每少 1 句 -30",
        "import re as _re",
        "from collections import Counter as _Counter",
        "",
        "_PUNCT = \"，。！？；、\"",
        "",
        "",
        "def _is_title_line(line):",
        '    return "《" in line or "》" in line',
        "",
        "",
        "def _is_author_line(line):",
        '    """独立成行、无句读、2~4 字的行视为作者行（模型输出格式: 标题行/作者行/诗句）"""',
        "    s = line.strip()",
        "    if not s or _is_title_line(s):",
        "        return False",
        "    if any(p in s for p in _PUNCT):",
        "        return False",
        "    return 2 <= len(s) <= 4",
        "",
        "",
        "def split_poem_lines(text):",
        '    """把模型生成文本拆成诗句列表（自动跳过标题行/作者行/空行）"""',
        "    poems = []",
        '    for raw in text.split("\\n"):',
        "        line = raw.strip()",
        "        if not line or _is_title_line(line) or _is_author_line(line):",
        "            continue",
        '        for seg in _re.split(r"[，。！？；、]", line):',
        "            seg = seg.strip()",
        "            if seg:",
        "                poems.append(seg)",
        "    return poems",
        "",
        "",
        "def score_poem(text):",
        '    """给一首五言诗打分，返回 (score, info)。',
        "",
        "    score 越高越好（满分 150 左右）;",
        "    info 含 lines/rhyme(韵部印记)/second_four_ok/dup/overflow/short 明细，供前端展示。",
        '    """',
        "    lines = split_poem_lines(text)",
        '    info = {"lines": len(lines), "rhyme": None, "second_four_ok": False,',
        '            "dup": 0, "overflow": 0, "short": 0, "title_in_rhyme": False}',
        "    score = 100.0",
        "",
        "    # 1) 严格五言",
        "    for ln in lines:",
        "        if len(ln) != 5:",
        '            info["short"] += 1',
        "            score -= 15",
        "",
        "    # 2) 二四句押韵",
        "    if len(lines) >= 4:",
        "        t2, t4 = lines[1][-1], lines[3][-1]",
        "        common = set(rhyme_of(t2)) & set(rhyme_of(t4))",
        "        if common:",
        '            info["second_four_ok"] = True',
        '            info["rhyme"] = common.pop()',
        "            score += 40",
        '            if info["rhyme"] in rhyme_of(lines[0][-1]):  # 首句入韵',
        '                info["title_in_rhyme"] = True',
        "                score += 10",
        "        else:",
        "            score -= 40",
        "    else:",
        "        score -= (4 - len(lines)) * 30",
        "",
        "    # 3) 句内重字",
        "    for ln in lines:",
        "        seen = set()",
        "        for ch in ln:",
        "            if ch in seen:",
        '                info["dup"] += 1',
        "                score -= 20",
        "            seen.add(ch)",
        "",
        "    # 4) 泛滥字",
        '    cnt = _Counter("".join(lines))',
        "    for ch, n in cnt.items():",
        "        if n > 3:",
        '            info["overflow"] += n - 3',
        "            score -= 10 * (n - 3)",
        "",
        "    return round(score, 1), info",
        "",
        "",
        "def evaluate_poem(text, meter=5):",
        '    """兼容旧接口：返回 {score, is_rhymed, rhyme_name, info}',
        "",
        "    poem.py / api_server.py 的 Best-of-N 重排调用此函数。",
        "    meter 参数保留（统一按五言考核；非五言按句扣分）。",
        '    """',
        "    score, info = score_poem(text)",
        "    return {",
        '        "score": score,',
        '        "is_rhymed": info["second_four_ok"],',
        '        "rhyme_name": info["rhyme"],',
        '        "info": info,',
        "    }",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已生成: {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
