# -*- coding: utf-8 -*-
"""抓取中华新韵（十四韵）韵字表并生成 rhyme_dict.py
数据源: 中华诗词学会《中华新韵》2005版 (华韵诗词学习网整理)
用途: 写诗模型 P0 押韵评分器
"""
import re
import sys
import urllib.request
from pathlib import Path

URL = "http://www.huayunonline.cn/hy-list-zhxy.htm"
OUT = Path(__file__).resolve().parent / "rhyme_dict.py"

# 尝试直连；失败则走系统代理/常见本地代理
def fetch(url, timeout=30):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return urllib.request.urlopen(req, timeout=timeout).read()
    except Exception as e:
        print(f"直连失败: {e}")
        # 走 Clash 常见端口
        for port in (7897, 7890, 10809):
            try:
                proxy = urllib.request.ProxyHandler({"http": f"http://127.0.0.1:{port}",
                                                     "https": f"http://127.0.0.1:{port}"})
                opener = urllib.request.build_opener(proxy)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                return opener.open(req, timeout=timeout).read()
            except Exception as e2:
                print(f"  代理 {port} 失败: {e2}")
    raise RuntimeError("所有网络通道均失败")

def decode(raw):
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")

def main():
    raw = fetch(URL)
    html = decode(raw)
    # 去掉 HTML 标签，保留文本
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;?", " ", text)

    # 韵部标题: 一麻 / 二波 / ... / 十四姑
    rhyme_titles = ["一麻", "二波", "三皆", "四开", "五微", "六豪", "七尤",
                    "八寒", "九文", "十唐", "十一庚", "十二齐", "十三支", "十四姑"]
    # 找到每个韵部标题出现的位置（标题后跟 【a，ia，ua】 这类韵母说明）
    positions = []
    for t in rhyme_titles:
        idx = text.find(t)
        if idx >= 0:
            positions.append((idx, t))
    positions.sort()
    if len(positions) < 14:
        print(f"只找到 {len(positions)} 个韵部标题，解析失败")
        print("前 500 字符:", text[:500])
        sys.exit(1)

    rhymes = {}
    for i, (idx, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        seg = text[idx:end]
        # 提取该韵部内所有汉字（含入声字，统一收进该韵部）
        chars = "".join(re.findall(r"[\u4e00-\u9fff]", seg))
        # 去掉标题本身的字（"一麻"等）
        for ch in title:
            chars = chars.replace(ch, "")
        # 去重（保持顺序）
        seen, dedup = set(), []
        for ch in chars:
            if ch not in seen:
                seen.add(ch)
                dedup.append(ch)
        rhymes[title] = "".join(dedup)
        print(f"{title}: {len(dedup)} 字")

    total = sum(len(v) for v in rhymes.values())
    print(f"共 {total} 字")

    # 反查表
    char_rhymes = {}
    for rhyme, chars in rhymes.items():
        for ch in chars:
            char_rhymes.setdefault(ch, []).append(rhyme)

    # 生成 rhyme_dict.py
    lines = [
        "# -*- coding: utf-8 -*-",
        '"""中华新韵（十四韵）韵字表 —— P0 押韵评分器数据',
        "数据来源: 中华诗词学会《中华新韵》(2005版)，华韵诗词学习网整理",
        "用法:",
        "  from rhyme_dict import rhyme_of, is_rhyme",
        "  rhyme_of('秋')   -> ['七尤']  (可跨多韵部: 多音字)",
        '  is_rhyme("愁", "楼")  -> True',
        '"""',
        "",
        "# 韵部 -> 韵字串（含派入的入声字，按韵部归类即可满足押韵判定）",
        "RHYME_CHARS = {",
    ]
    for title, chars in rhymes.items():
        lines.append(f'    "{title}": "{chars}",')
    lines.append("}")
    lines += [
        "",
        "# 反查: 字 -> 所属韵部列表（多音字可能跨韵部）",
        "CHAR_RHYMES = {}",
        "for _r, _cs in RHYME_CHARS.items():",
        "    for _c in _cs:",
        "        CHAR_RHYMES.setdefault(_c, []).append(_r)",
        "",
        "",
        "def rhyme_of(ch):",
        '    """返回某字的韵部列表；生僻字/词表外返回空列表"""',
        "    return CHAR_RHYMES.get(ch, [])",
        "",
        "",
        "def is_rhyme(ch1, ch2):",
        '    """两字是否可押韵（所属韵部有交集）"""',
        "    r1 = CHAR_RHYMES.get(ch1, [])",
        "    r2 = CHAR_RHYMES.get(ch2, [])",
        "    return bool(set(r1) & set(r2))",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已生成: {OUT}")


if __name__ == "__main__":
    main()
