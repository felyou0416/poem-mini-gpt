# -*- coding: utf-8 -*-
"""
v3 写诗模型训练：MiniGPT_v3（现代小模型架构）
===========================================
架构升级（vs v2 的 4层/128维/LayerNorm/绝对位置）:
  - 6 层 / 192 维 / 6 头（每头 32 维）
  - RoPE 旋转位置编码（替代绝对位置 Embedding，天然捕捉五言 2-3 停顿韵律）
  - RMSNorm（替代 LayerNorm，省去均值中心化，训练提速）
  - 词表 4006（4000 高频字符 + 6 特殊 token：bos/title/author/body/eos/unk）
  - 数据格式: <|bos|>《题》<|title|>作者<|author|><|body|>诗句<|eos|>
    生成遇 <|eos|> 自动停机（P3 特殊 token 控制，训练端已就绪）

训练经验（小模型先进方法）:
  - WSD 学习率调度（预热→稳定→最后 10% 线性衰减），可随时断点续训
  - 92,063 首严格五言（每句恰好 5 字），清洗+去重，数据质量优先
  - 多 epoch：711 万 token / (batch32×128) ≈ 1736 步一个 epoch，6000 步 ≈ 3.5 epoch

用法: python train_v3.py [总步数]
输出: 五言BPE_v3\\wuyan_bpe_v3_model.pt / loss_log.csv / loss_curve.png / train_state.json
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

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "五言BPE_v3"
PREFIX = "wuyan_bpe_v3_"

# ---------- 超参数 ----------
N_LAYER = 6
N_HEAD = 6
N_EMBD = 192
DROPOUT = 0.1
LR = 5e-4
BATCH_SIZE = 32
MAX_STEPS = 6000
EVAL_INTERVAL = 100
SAVE_INTERVAL = 500
WARMUP_STEPS = 500
DECAY_FRACTION = 0.1   # WSD：最后 10% 步数线性衰减到 0
GRAD_CLIP = 1.0
ROPE_BASE = 10000.0

torch.set_num_threads(max(1, __import__("os").cpu_count() or 4))


# ---------- 数据加载 ----------
def load_data(name: str):
    raw = np.fromfile(DATA_DIR / f"{PREFIX}{name}.bin", dtype=np.uint16)
    return torch.from_numpy(raw.astype(np.int64))


def get_batch(split, data, block_size, batch_size, device):
    d = data[split]
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------- 现代架构组件 ----------
class RMSNorm(nn.Module):
    """RMS 归一化：去掉均值中心化，速度更快、效果相当（Llama 系列标配）"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RoPE(nn.Module):
    """旋转位置编码：把位置信息以旋转矩阵注入 q/k，天然支持相对位置（韵律周期）
    实现采用 Llama/NeoX 标准 half-split：cos/sin 各 D/2 维，分别旋转前/后半维
    """
    def __init__(self, head_dim, max_seq_len, base=ROPE_BASE):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                      # (T, D/2)
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))

    def apply(self, q, k, T):
        cos = self.cos_cached[:, :, :T]                       # 1,1,T,D/2
        sin = self.sin_cached[:, :, :T]
        d = q.shape[-1] // 2
        q1, q2 = q[..., :d], q[..., d:]
        k1, k2 = k[..., :d], k[..., d:]
        q = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
        k = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.rope = RoPE(self.head_dim, block_size)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rope.apply(q, k, T)                        # RoPE 注入位置
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
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT_v3(nn.Module):
    """v3 现代架构：RoPE + RMSNorm + 深度扩容（无绝对位置嵌入）"""
    def __init__(self, vocab_size, block_size, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # 权重共享

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)                        # 位置信息由 RoPE 在注意力内注入
        x = self.blocks(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------- 断点续训 ----------
def load_checkpoint_progress(model, max_steps):
    state_path = DATA_DIR / f"{PREFIX}train_state.json"
    model_path = DATA_DIR / f"{PREFIX}model.pt"
    if not model_path.exists():
        return 0
    step0 = 0
    if state_path.exists():
        try:
            step0 = int(json.loads(state_path.read_text(encoding="utf-8"))["step"])
        except Exception:
            step0 = 0
    if step0 <= 0:
        log_path = DATA_DIR / f"{PREFIX}loss_log.csv"
        if log_path.exists():
            n = sum(1 for _ in open(log_path, encoding="utf-8")) - 1
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
    # 兼容 keep_training 传入 [prefix, steps] 与直接调用 [steps]
    max_steps = int(sys.argv[-1]) if len(sys.argv) > 1 else MAX_STEPS
    meta = json.loads((DATA_DIR / f"{PREFIX}meta.json").read_text(encoding="utf-8-sig"))
    vocab_size = meta["vocab_size"]
    block_size = meta["block_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    data = {s: load_data(s) for s in ("train", "val")}

    model = MiniGPT_v3(vocab_size=vocab_size, block_size=block_size)
    model.to(device)
    print(f"设备: {device} | 参数量: {model.count_params():,} | 词表: {vocab_size} | block: {block_size}")
    print(f"架构: {N_LAYER}层/{N_EMBD}维/{N_HEAD}头 RoPE+RMSNorm | 训练 {max_steps} 步")

    xb, yb = get_batch("train", data, block_size, 2, device)
    _, loss0 = model(xb, yb)
    print(f"forward 检查通过, 初始 loss ≈ {loss0.item():.4f} (期望 ≈ ln(vocab)={math.log(vocab_size):.2f})")

    step0 = load_checkpoint_progress(model, max_steps)
    if step0 >= max_steps:
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
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

    log_path = DATA_DIR / f"{PREFIX}loss_log.csv"
    if step0 == 0:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss"])

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "step": step, "config": {
            "vocab_size": vocab_size, "block_size": block_size,
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
            "dropout": DROPOUT, "arch": "v3"}}, DATA_DIR / f"{PREFIX}model.pt")
        (DATA_DIR / f"{PREFIX}train_state.json").write_text(json.dumps({"step": step}), encoding="utf-8")

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
            print(f"  [保存] {PREFIX}model.pt @ step {step}")

    print(f"训练完成, 耗时 {(time.time()-t0)/60:.1f} 分钟, 最终模型: {PREFIX}model.pt")

    # 画 loss 曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
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
        plt.title("v3 写诗模型训练曲线（6层/192维 RoPE+RMSNorm）")
        plt.tight_layout()
        plt.savefig(DATA_DIR / f"{PREFIX}loss_curve.png", dpi=130)
        print(f"loss 曲线已保存: {PREFIX}loss_curve.png")
    except Exception as e:
        print(f"[!] 画图失败(不影响模型): {e}")


if __name__ == "__main__":
    main()
