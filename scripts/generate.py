# -*- coding: utf-8 -*-
"""
用训练好的写诗模型生成诗歌
用法: python generate.py [开头文本] [温度] [数据前缀]
示例: python generate.py "《春日》" 0.8 wuyan_   # 用五言模型
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent   # 写诗模型根目录（脚本在 scripts/ 下）


def data_dir(prefix: str) -> Path:
    """按数据前缀映射到对应子目录"""
    mapping = {
        "": ROOT / "通用模型",
        "wuyan_": ROOT / "归档\五言数据",
        "wuyan_bpe_": ROOT / "五言BPE",
        "jueju_": ROOT / "绝句模型",
    }
    return mapping.get(prefix, ROOT)


def load_model(prefix=""):
    ckpt = torch.load(data_dir(prefix) / f"{prefix}model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    itos = json.loads((data_dir(prefix) / f"{prefix}vocab.json").read_text(encoding="utf-8-sig"))
    itos = {int(k): v for k, v in itos.items()}  # JSON 键是字符串，转回整数
    stoi = {c: i for i, c in itos.items()}

    from train import MiniGPT
    model = MiniGPT(vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                    dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, stoi, itos, cfg


@torch.no_grad()
def generate(model, stoi, itos, cfg, seed="《春日》", length=100, temperature=0.8, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    # 把 seed 编码（只保留词表内字符）
    chars = [c for c in seed if c in stoi]
    if not chars:
        chars = list(seed)[:1]
    idx = torch.tensor([stoi[c] for c in chars], dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(length):
        idx_cond = idx if idx.size(1) <= cfg["block_size"] else idx[:, -cfg["block_size"]:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-3)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)

    return "".join(itos[int(i)] for i in idx[0])


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else "《春日》"
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""

    model, stoi, itos, cfg = load_model(prefix)
    print(f"模型加载完成({prefix or '默认'}): {cfg['n_layer']}层/{cfg['n_head']}头/{cfg['n_embd']}维, block={cfg['block_size']}")

    examples = []
    for temp in ([temperature] if len(sys.argv) > 2 else [0.6, 0.8, 1.0]):
        out = generate(model, stoi, itos, cfg, seed=seed, length=100, temperature=temp)
        examples.append(f"--- 温度 {temp} ---\n{out}\n")
        print(f"\n=== 温度 {temp} ===")
        print(out)

    (data_dir(prefix) / f"{prefix}generated_examples.txt").write_text(
        f"seed: {seed}\n" + "\n".join(examples), encoding="utf-8")
    print(f"\n样例已保存: {prefix}generated_examples.txt")


if __name__ == "__main__":
    main()
