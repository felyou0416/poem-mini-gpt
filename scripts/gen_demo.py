# -*- coding: utf-8 -*-
"""批量生成多组五言绝句样例（多个开头 + 温度），汇总保存"""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
seeds = ["《春日》", "《夜雨》", "《秋思》", "《归隐》"]
temps = (0.6, 0.8)

parts = [f"五言绝句模型生成样例（模型: jueju_model.pt, 2000 步, val loss 4.72）", ""]
for s in seeds:
    for t in temps:
        r = subprocess.run(
            [sys.executable, str(BASE / "generate.py"), s, str(t), "jueju_"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = r.stdout
        if "=== 温度" in out:
            body = out.split("=== 温度", 1)[1].split("样例已保存", 1)[0].strip()
        else:
            body = out.strip()
        parts.append(f"=== 开头「{s}」· 温度 {t} ===")
        parts.append(body)
        parts.append("")

(BASE / "jueju_generated_examples.txt").write_text("\n".join(parts), encoding="utf-8")
print("已保存: jueju_generated_examples.txt")
