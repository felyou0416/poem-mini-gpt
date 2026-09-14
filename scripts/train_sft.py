# -*- coding: utf-8 -*-
"""
P2 名篇 SFT 微调：以 v3 预训练模型为起点，注入盛唐雅韵
====================================================
数据: 语料/sft_tang_corpus.txt（3,125 首 李杜王孟等名家严格五言）
起点: 五言BPE_v3/wuyan_bpe_v3_model.pt（P1c 预训练 checkpoint）
防过拟合（方案 P2）:
  - 学习率 5e-5（预训练 5e-4 的 1/10）
  - 步数 600（严格限制，避免机械背诵原诗）
  - Dropout 0.15（预训练 0.1 → 提高防呆）
  - 早停：val loss 连续 3 次评估不降则提前结束
输出: 五言BPE_v3/wuyan_bpe_v3_sft_model.pt（不覆盖预训练模型）
用法: python train_sft.py [总步数]
"""
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_v3 import MiniGPT_v3, DATA_DIR, PREFIX

ROOT = Path(__file__).resolve().parent.parent
SFT_TXT = ROOT / "语料" / "sft_tang_corpus.txt"
PRETRAIN_CKPT = DATA_DIR / f"{PREFIX}model.pt"
OUT_CKPT = DATA_DIR / "wuyan_bpe_v3_sft_model.pt"
OUT_STATE = DATA_DIR / "wuyan_bpe_v3_sft_train_state.json"
OUT_LOG = DATA_DIR / "wuyan_bpe_v3_sft_loss_log.csv"

# ---------- SFT 超参数（方案 P2：防死记硬背） ----------
LR = 5e-5
BATCH_SIZE = 32
MAX_STEPS = 600
DROPOUT = 0.15
EVAL_INTERVAL = 50
SAVE_INTERVAL = 100
EARLY_STOP_ROUNDS = 3     # val 连续 3 次不降 → 早停
WARMUP_STEPS = 50
GRAD_CLIP = 1.0
VAL_FRACTION = 0.10
BLOCK_SIZE = 128

_PUNCT = "，。！？；、"


def load_sft_data():
    """sft 语料 → 用 v3 词表编码成特殊 token 序列（与预训练格式一致）"""
    meta = json.loads((DATA_DIR / f"{PREFIX}meta.json").read_text(encoding="utf-8-sig"))
    itos = json.loads((DATA_DIR / f"{PREFIX}vocab.json").read_text(encoding="utf-8-sig"))
    itos = {int(k): v for k, v in itos.items()}
    stoi = {c: i for i, c in itos.items()}
    unk = stoi.get("<|unk|>", 5)

    text = SFT_TXT.read_text(encoding="utf-8")
    blocks = [b for b in text.split("\n\n") if b.strip()]

    encoded = []
    for b in blocks:
        lines = [l.strip() for l in b.split("\n") if l.strip()]
        if not lines:
            continue
        title, author, body = None, None, []
        for ln in lines:
            if "《" in ln or "》" in ln:
                title = ln
            elif title is not None and author is None and len(ln) <= 4 and not any(p in ln for p in _PUNCT):
                author = ln
            else:
                segs = [s for s in re.split(r"[，。！？；、]", ln.replace(" ", "")) if s.strip()]
                if segs and all(len(s) == 5 for s in segs):
                    body.append(ln.replace(" ", ""))
        if title is None or author is None or not body:
            continue
        seq = [0] + [stoi.get(c, unk) for c in title] + [1] + [stoi.get(c, unk) for c in author] + [2, 3]
        seq += [stoi.get(c, unk) for c in "".join(body)[:100]]
        seq += [4]   # <|eos|>
        encoded.append(seq)

    print(f"SFT 语料编码: {len(encoded):,} 首")
    return encoded, meta["vocab_size"], meta["block_size"]


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def main():
    max_steps = int(sys.argv[-1]) if len(sys.argv) > 1 else MAX_STEPS
    encoded, vocab_size, block_size = load_sft_data()

    rng = np.random.default_rng(42)
    n_val = max(1, int(len(encoded) * VAL_FRACTION))
    idx = rng.permutation(len(encoded))
    def flat(bs):
        out = []
        for e in bs:
            out.extend(e)
        return torch.tensor(out, dtype=torch.long)
    data = {"train": flat([encoded[i] for i in idx[n_val:]]),
            "val": flat([encoded[i] for i in idx[:n_val]])}
    print(f"SFT 训练 token: {len(data['train']):,} | val token: {len(data['val']):,}")

    # 从预训练 checkpoint 加载（修改 dropout）
    ckpt = torch.load(PRETRAIN_CKPT, map_location="cpu", weights_only=False)
    model = MiniGPT_v3(vocab_size=vocab_size, block_size=block_size, dropout=DROPOUT)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"已加载预训练 checkpoint（{PRETRAIN_CKPT.name}）→ 参数量 {model.count_params():,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    torch.manual_seed(1337)

    xb, yb = get_batch(data["train"], block_size, 2)
    _, loss0 = model(xb, yb)
    print(f"forward 检查通过, 初始 loss ≈ {loss0.item():.4f}")

    # 断点续训（SFT 自己的 state 文件）
    step0 = 0
    if OUT_STATE.exists():
        try:
            step0 = int(json.loads(OUT_STATE.read_text(encoding="utf-8"))["step"])
            if step0 > 0:
                sckpt = torch.load(OUT_CKPT, map_location="cpu", weights_only=False)
                model.load_state_dict(sckpt["model"])
                print(f"从 SFT checkpoint 恢复: 已训 {step0} 步")
        except Exception:
            step0 = 0
    if step0 >= max_steps:
        print(f"SFT 已完成 {step0}/{max_steps} 步")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)

    def lr_at(step):
        if step < WARMUP_STEPS:
            return LR * step / max(1, WARMUP_STEPS)
        return LR * max(0.0, 1 - (step - WARMUP_STEPS) / max(1.0, max_steps - WARMUP_STEPS))  # 余弦式衰减到 0

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(10):
            xb, yb = get_batch(data["val"], block_size, BATCH_SIZE)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    if step0 == 0:
        with open(OUT_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss"])

    best_val, stale = float("inf"), 0
    t0 = time.time()
    print(f"开始 SFT 微调({max_steps} 步, LR {LR}, Dropout {DROPOUT})...")
    for step in range(step0 + 1, max_steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        xb, yb = get_batch(data["train"], block_size, BATCH_SIZE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == 1:
            val_loss = evaluate() if step > 1 else None
            with open(OUT_LOG, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{val_loss:.4f}" if val_loss else ""])
            val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            print(f"step {step}/{max_steps} | loss {loss.item():.4f} | val {val_str} | lr {lr_at(step):.2e} | {(time.time()-t0):.0f}s")
            # 早停
            if val_loss is not None:
                if val_loss < best_val - 1e-3:
                    best_val, stale = val_loss, 0
                else:
                    stale += 1
                    if stale >= EARLY_STOP_ROUNDS:
                        print(f"早停: val 连续 {EARLY_STOP_ROUNDS} 次未下降，提前结束（保留最佳权重）")
                        max_steps = step
                        break

        if step % SAVE_INTERVAL == 0 or step == max_steps:
            torch.save({"model": model.state_dict(), "step": step, "config": {
                "vocab_size": vocab_size, "block_size": block_size,
                "n_layer": 6,
                "n_head": model.blocks[0].attn.n_head,
                "n_embd": model.n_embd, "dropout": DROPOUT, "arch": "v3", "sft": True}},
                OUT_CKPT)
            OUT_STATE.write_text(json.dumps({"step": step}), encoding="utf-8")
            print(f"  [保存] {OUT_CKPT.name} @ step {step}")

    print(f"SFT 完成, 耗时 {(time.time()-t0)/60:.1f} 分钟, 最终模型: {OUT_CKPT.name}（预训练模型未动）")


if __name__ == "__main__":
    main()
