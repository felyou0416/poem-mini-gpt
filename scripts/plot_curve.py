# -*- coding: utf-8 -*-
"""读取 loss_log.csv 重画 loss 曲线（修复中文字体）
用法: python plot_curve.py [数据前缀]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent   # 写诗模型根目录（脚本在 scripts/ 下）


def data_dir(prefix: str) -> Path:
    """按数据前缀映射到对应子目录"""
    mapping = {
        "": ROOT / "通用模型",
        "wuyan_": ROOT / "归档\五言数据",
        "wuyan_bpe_": ROOT / "五言BPE",
        "wuyan_bpe_v3_": ROOT / "五言BPE_v3",
        "jueju_": ROOT / "绝句模型",
    }
    return mapping.get(prefix, ROOT)


prefix = sys.argv[1] if len(sys.argv) > 1 else ""
log_path = data_dir(prefix) / f"{prefix}loss_log.csv"

steps, tl, vl = [], [], []
with open(log_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["train_loss"]:
            steps.append(int(row["step"]))
            tl.append(float(row["train_loss"]))
            if row["val_loss"]:
                vl.append((int(row["step"]), float(row["val_loss"])))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, tl, label="训练 loss", linewidth=2, color="#0e7c66")
if vl:
    ax.plot([p[0] for p in vl], [p[1] for p in vl], "o-", label="验证 loss",
            markersize=4, color="#e0a33c")
ax.set_xlabel("训练步数 (step)")
ax.set_ylabel("loss")
ax.set_title("五言绝句模型训练曲线" if prefix == "jueju_" else f"{prefix}模型训练曲线")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(data_dir(prefix) / f"{prefix}loss_curve.png", dpi=130)
print(f"已保存: {prefix}loss_curve.png")
