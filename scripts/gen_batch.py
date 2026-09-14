# -*- coding: utf-8 -*-
"""③ 批量生成：多开头 × 多温度，探索绝句模型的不同效果
用法: python gen_batch.py
"""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent

# 开头覆盖季节/景物/情感，温度覆盖低(稳)/中/高(放飞)
seeds = ["《春日》", "《夜雨》", "《秋思》", "《归隐》", "《咏梅》", "《江行》", "《山居》", "《送别》"]
temps = (0.4, 0.6, 0.8, 1.0)

parts = ["五言绝句模型·批量生成实验（jueju_model.pt, 2000 步）", ""]
for s in seeds:
    for t in temps:
        r = subprocess.run(
            [sys.executable, str(BASE / "generate.py"), s, str(t), "jueju_"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = r.stdout
        if "=== 温度" in out:
            body = out.split("=== 温度", 1)[1].split("样例已保存", 1)[0].strip()
            # 去掉首行残留的 "x ===" 
            lines = body.split("\n")
            if lines and lines[0].strip().rstrip("=").strip().isdigit() is False and "===" in lines[0]:
                lines = lines[1:]
            body = "\n".join(lines).strip()
        else:
            body = out.strip()
        parts.append(f"=== 「{s}」· 温度 {t} ===")
        parts.append(body)
        parts.append("")

out_file = BASE / "jueju_batch_examples.txt"
out_file.write_text("\n".join(parts), encoding="utf-8")
print(f"已保存 {len(seeds) * len(temps)} 组样例: {out_file.name}")
