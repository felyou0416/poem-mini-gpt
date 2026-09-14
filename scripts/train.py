# -*- coding: utf-8 -*-
"""
训练一个迷你 GPT 写诗模型（CPU 可跑）
用法: python train.py [数据前缀] [步数]
示例: python train.py wuyan_       # 读 wuyan_meta.json 等，输出 wuyan_model.pt
      python train.py jueju_ 1000  # 绝句版，只训 1000 步
模型: ~250 万参数小 Transformer，字符级自回归
输出: {prefix}model.pt (checkpoint), {prefix}loss_log.csv, {prefix}loss_curve.png
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent   # 写诗模型根目录（脚本统一放在 scripts/ 下）


def data_dir(prefix: str) -> Path:
    """按数据前缀映射到对应子目录"""
    mapping = {
        "": ROOT / "通用模型",
        "wuyan_": ROOT / "归档\五言数据",
        "wuyan_bpe_": ROOT / "五言BPE",
        "jueju_": ROOT / "绝句模型",
    }
    return mapping.get(prefix, ROOT)

# ---------- 超参数 ----------
# 参考小模型先进训练经验（TinyLlama / SmolLM3 / MiniCPM）：
#   - WSD 学习率调度（预热→稳定→最后 10% 线性衰减），可随时续训
#   - 学习率 5e-4（小模型惯例 2e-4~4e-4 附近）
#   - 训练多遍数据（小模型应远超 1 epoch）
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1
LR = 5e-4
BATCH_SIZE = 32
MAX_STEPS = 5000
EVAL_INTERVAL = 100
SAVE_INTERVAL = 500
WARMUP_STEPS = 500
DECAY_FRACTION = 0.1   # WSD：最后 10% 步数线性衰减到 0
GRAD_CLIP = 1.0

# ---------- 数据加载 ----------
torch.set_num_threads(max(1, __import__("os").cpu_count() or 4))


def load_data(name: str, prefix: str = ""):
    raw = np.fromfile(data_dir(prefix) / f"{prefix}{name}.bin", dtype=np.uint16)
    return torch.from_numpy(raw.astype(np.int64))


def get_batch(split: str, data: dict, block_size: int, batch_size: int, device):
    d = data[split]
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------- 模型 ----------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.g * (x - mean) / torch.sqrt(var + self.eps) + self.b


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # 权重共享

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------- 断点续训 ----------
def load_checkpoint_progress(prefix: str, model: nn.Module, max_steps: int):
    """
    检查已有 checkpoint，返回应从哪一步继续训练 (step0)。
    step0 == 0 表示从头训练。支持两种进度来源：
      1. {prefix}train_state.json（本脚本保存的 step）
      2. 旧版 {prefix}model.pt 无 step 字段时，用 loss_log.csv 行数×EVAL_INTERVAL 估算
    """
    state_path = data_dir(prefix) / f"{prefix}train_state.json"
    model_path = data_dir(prefix) / f"{prefix}model.pt"
    if not model_path.exists():
        return 0
    step0 = 0
    if state_path.exists():
        try:
            step0 = int(json.loads(state_path.read_text(encoding="utf-8"))["step"])
        except Exception:
            step0 = 0
    if step0 <= 0:
        log_path = data_dir(prefix) / f"{prefix}loss_log.csv"
        if log_path.exists():
            n = sum(1 for _ in open(log_path, encoding="utf-8")) - 1  # 去掉表头
            step0 = max(0, n * EVAL_INTERVAL)
    if step0 >= max_steps:
        print(f"checkpoint 已训练到 {step0}/{max_steps} 步, 无需再训")
        return max_steps
    if step0 > 0:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"从 checkpoint 恢复: 已训 {step0} 步, 继续训练到 {max_steps} 步")
    return step0


# ---------- 主流程 ----------
def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_STEPS
    meta = json.loads((data_dir(prefix) / f"{prefix}meta.json").read_text(encoding="utf-8-sig"))
    vocab_size = meta["vocab_size"]
    block_size = meta["block_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    data = {s: load_data(s, prefix) for s in ("train", "val")}

    model = MiniGPT(vocab_size=vocab_size, block_size=block_size)
    model.to(device)
    print(f"设备: {device} | 参数量: {model.count_params():,} | 词表: {vocab_size} | block: {block_size}")

    # 用一小批数据先验证 forward 没问题
    xb, yb = get_batch("train", data, block_size, 2, device)
    _, loss0 = model(xb, yb)
    print(f"forward 检查通过, 初始 loss ≈ {loss0.item():.4f} (期望 ≈ ln(vocab)={math.log(vocab_size):.2f})")

    step0 = load_checkpoint_progress(prefix, model, max_steps)
    if step0 >= max_steps:
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        """WSD 学习率调度：预热线性升 → 稳定期恒定 → 最后 10% 线性衰减到 0"""
        if step < WARMUP_STEPS:
            return LR * step / max(1, WARMUP_STEPS)
        decay_start = max_steps * (1 - DECAY_FRACTION)
        if step >= decay_start:
            return LR * max(0.0, (max_steps - step) / max(1.0, max_steps - decay_start))
        return LR

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(20):
            xb, yb = get_batch("val", data, block_size, BATCH_SIZE, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    log_path = data_dir(prefix) / f"{prefix}loss_log.csv"
    if step0 == 0:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss"])

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "step": step, "config": {
            "vocab_size": vocab_size, "block_size": block_size,
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD, "dropout": DROPOUT}}, data_dir(prefix) / f"{prefix}model.pt")
        (data_dir(prefix) / f"{prefix}train_state.json").write_text(
            json.dumps({"step": step}), encoding="utf-8")

    print(f"开始训练({max_steps} 步)...")
    t0 = time.time()
    for step in range(step0 + 1, max_steps + 1):
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
            el = time.time() - t0
            val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            print(f"step {step}/{max_steps} | loss {loss.item():.4f} | val {val_str} | lr {lr_at(step):.2e} | {el:.0f}s")

        if step % SAVE_INTERVAL == 0 or step == max_steps:
            save_ckpt(step)
            print(f"  [保存] {prefix}model.pt @ step {step}")

    print(f"训练完成, 耗时 {(time.time()-t0)/60:.1f} 分钟, 最终模型: {prefix}model.pt")

    # 画 loss 曲线
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
        plt.plot(steps, tl, label="train loss")
        if vl:
            plt.plot([p[0] for p in vl], [p[1] for p in vl], "o-", label="val loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.title("写诗小模型训练曲线")
        plt.tight_layout()
        plt.savefig(data_dir(prefix) / f"{prefix}loss_curve.png", dpi=130)
        print(f"loss 曲线已保存: {prefix}loss_curve.png")
    except Exception as e:
        print(f"[!] 画图失败(不影响模型): {e}")


if __name__ == "__main__":
    main()
