# -*- coding: utf-8 -*-
"""
二次训练（续训）：加载 model.pt，用更低学习率继续训练
输入: model.pt
输出: model_ft.pt (并覆盖为 model.pt 作为最新模型), loss_log_ft.csv, loss_curve_ft.png
"""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from train import MiniGPT, get_batch, load_data

ROOT = Path(__file__).resolve().parent.parent   # 写诗模型根目录（脚本在 scripts/ 下）
MODEL_DIR = ROOT / "通用模型"

# ---------- 二次训练超参（更低学习率，精细优化） ----------
LR = 3e-4
BATCH_SIZE = 32
MAX_STEPS = 2000
EVAL_INTERVAL = 100
SAVE_INTERVAL = 500
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
CKPT_IN = MODEL_DIR / "model.pt"
CKPT_OUT = MODEL_DIR / "model_ft.pt"


def main():
    ckpt = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2026)

    model = MiniGPT(vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                    dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"已加载 {CKPT_IN.name} | 参数 {sum(p.numel() for p in model.parameters()):,} | 设备 {device}")

    data = {s: load_data(s, "") for s in ("train", "val")}
    block_size = cfg["block_size"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < WARMUP_STEPS:
            return LR * step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return LR * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(15):
            xb, yb = get_batch("val", data, block_size, BATCH_SIZE, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    log_path = MODEL_DIR / "loss_log_ft.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss"])

    print(f"开始二次训练（续训 {MAX_STEPS} 步, lr={LR}）...")
    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        xb, yb = get_batch("train", data, block_size, BATCH_SIZE, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == 1:
            val_loss = evaluate() if step > 1 else None
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{val_loss:.4f}" if val_loss else ""])
            val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            print(f"step {step}/{MAX_STEPS} | loss {loss.item():.4f} | val {val_str} | lr {lr_at(step):.2e} | {(time.time()-t0):.0f}s")

        if step % SAVE_INTERVAL == 0:
            torch.save({"model": model.state_dict(), "config": cfg}, CKPT_OUT)
            print(f"  [保存] {CKPT_OUT.name} @ step {step}")

    torch.save({"model": model.state_dict(), "config": cfg}, CKPT_OUT)
    # 覆盖为最新模型，generate.py 默认加载它
    import shutil
    shutil.copyfile(CKPT_OUT, MODEL_DIR / "model.pt")
    print(f"二次训练完成, 耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"最终模型: {CKPT_OUT.name}（已同步为 model.pt）")

    # loss 曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, tl, vl = [], [], []
        with open(log_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["train_loss"]:
                    steps.append(int(row["step"]))
                    tl.append(float(row["train_loss"]))
                    if row["val_loss"]:
                        vl.append((int(row["step"]), float(row["val_loss"])))
        plt.figure(figsize=(8, 5))
        plt.plot(steps, tl, label="train loss (续训)")
        if vl:
            plt.plot([p[0] for p in vl], [p[1] for p in vl], "o-", label="val loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.title("写诗模型二次训练曲线")
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "loss_curve_ft.png", dpi=130)
        print("二次训练曲线已保存: loss_curve_ft.png")
    except Exception as e:
        print(f"[!] 画图失败: {e}")


if __name__ == "__main__":
    main()
