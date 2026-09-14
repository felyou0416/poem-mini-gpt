# -*- coding: utf-8 -*-
"""
export_android.py — 把 v3·SFT 写诗模型导出为安卓 APP 可加载格式

输出到 安卓APP/app/src/main/assets/:
  model.bin        全部权重 float32 顺序拼接
  manifest.json    key -> {offset(float32索引), shape}
  vocab.json       词表 itos（拷贝自 v3 目录）
  meta.json        超参 + 特殊token + 采样参数
  rhyme.json       平水韵十四韵字表
  selftest.json    固定输入 -> top20 logits（安卓端启动自检用）

用法: python export_android.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent          # 写诗模型/
ASSET_DIR = ROOT / "安卓APP" / "app" / "src" / "main" / "assets"
V3_DIR = ROOT / "五言BPE_v3"
CKPT = V3_DIR / "wuyan_bpe_v3_sft_model.pt"
VOCAB = V3_DIR / "wuyan_bpe_v3_vocab.json"

sys.path.insert(0, str(ROOT / "scripts"))
from train_v3 import MiniGPT_v3  # noqa: E402
import rhyme_dict  # noqa: E402


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 加载 checkpoint
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    cfg = ckpt.get("config", {})
    print(f"[导出] checkpoint: {CKPT.name} | step={ckpt.get('step')}")
    print(f"[导出] config: {cfg}")

    # 2) 权重 -> model.bin + manifest.json
    offsets = {}
    buf = []
    for key, t in sd.items():
        t = t.detach().cpu().float().numpy().reshape(-1)
        offsets[key] = {"offset": len(buf), "shape": list(tuple(sd[key].shape))}
        buf.append(t)
    all_w = np.concatenate(buf).astype(np.float32)
    all_w.tofile(ASSET_DIR / "model.bin")
    (ASSET_DIR / "manifest.json").write_text(
        json.dumps(offsets, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[导出] model.bin: {len(all_w):,} floats = {len(all_w)*4/1e6:.1f} MB, {len(offsets)} 项权重")

    # 3) vocab.json 拷贝
    vocab = json.loads(VOCAB.read_text(encoding="utf-8-sig"))
    vocab_int = {int(k): v for k, v in vocab.items()}
    (ASSET_DIR / "vocab.json").write_text(
        json.dumps(vocab_int, ensure_ascii=False), encoding="utf-8")
    print(f"[导出] vocab.json: {len(vocab_int)} 词元")

    # 4) meta.json
    special = {"bos": 0, "title": 1, "author": 2, "body": 3, "eos": 4, "unk": 5}
    meta = {
        "name": "五言BPE v3·SFT",
        "arch": "v3",
        "n_layer": cfg.get("n_layer", 6),
        "n_embd": cfg.get("n_embd", 192),
        "n_head": cfg.get("n_head", 6),
        "block_size": cfg.get("block_size", 128),
        "vocab_size": cfg.get("vocab_size", 4006),
        "dropout": 0.0,             # 推理期关闭
        "norm_eps": 1e-6,
        "rope_base": 10000.0,
        "special": special,
        "sampling": {"temperature": 0.6, "top_k": 60, "max_tokens": 140,
                     "candidates_k": 12, "num_poems": 3},
    }
    (ASSET_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[导出] meta.json: {meta['n_layer']}层/{meta['n_embd']}维/{meta['n_head']}头")

    # 5) rhyme.json
    rhyme = {"order": rhyme_dict.RHYME_ORDER,
             "chars": {r: list(rhyme_dict.RHYME_CHARS[r]) for r in rhyme_dict.RHYME_ORDER}}
    (ASSET_DIR / "rhyme.json").write_text(
        json.dumps(rhyme, ensure_ascii=False), encoding="utf-8")
    total_chars = sum(len(v) for v in rhyme["chars"].values())
    print(f"[导出] rhyme.json: {len(rhyme['order'])} 韵部 / {total_chars} 字")

    # 6) selftest.json —— 固定输入 forward 的 top20 logits（安卓端启动自检）
    model = MiniGPT_v3(vocab_size=meta["vocab_size"], block_size=meta["block_size"])
    model.load_state_dict(sd)
    model.eval()
    stoi = {c: i for i, c in vocab_int.items()}   # 反向: 字 -> id
    seed_tokens = [special["bos"]] + [stoi[k] for k in "山高水流"] + [special["title"]]
    x = torch.tensor([seed_tokens], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
    logits = logits[0, -1, :]
    topv, topi = torch.topk(logits, 20)
    selftest = {
        "tokens": seed_tokens,
        "top20_ids": [int(i) for i in topi.tolist()],
        "top20_logits": [round(float(v), 4) for v in topv.tolist()],
    }
    (ASSET_DIR / "selftest.json").write_text(
        json.dumps(selftest, ensure_ascii=False), encoding="utf-8")
    print(f"[导出] selftest.json: 输入 {seed_tokens} -> top1 id={selftest['top20_ids'][0]} "
          f"({vocab_int[selftest['top20_ids'][0]]}) logit={selftest['top20_logits'][0]}")

    print(f"[导出] 完成 → {ASSET_DIR}")


if __name__ == "__main__":
    main()
