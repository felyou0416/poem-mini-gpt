# -*- coding: utf-8 -*-
"""
一键写诗工具（支持 3 个模型切换）
温度已默认 0.6（批量实验得出的最佳平衡点：0.4 太呆、0.8+ 易乱）
用法: python poem.py [开头] [温度] [数量] [模型id]
模型id: wuyan_bpe（默认，当前最强）/ jueju / all
示例: python poem.py 秋思                # 五言BPE v2、默认温度 0.6、出 3 首
      python poem.py 咏梅 0.8 5 jueju    # 用绝句模型
"""
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent   # 写诗模型根目录
# 模型注册表：id -> (数据前缀, 显示名, 目录名, checkpoint文件名[可选])
MODELS = {
    "wuyan_bpe_v3":    ("wuyan_bpe_v3_",    "五言BPE v3",     "五言BPE_v3"),
    "wuyan_bpe_v3_sft": ("wuyan_bpe_v3_",   "五言BPE v3·SFT", "五言BPE_v3", "wuyan_bpe_v3_sft_model.pt"),
    "wuyan_bpe":       ("wuyan_bpe_",       "五言BPE v2",     "五言BPE"),
    "jueju":           ("jueju_",           "五言绝句",       "绝句模型"),
    "all":             ("",                 "通用模型",       "通用模型"),
}
DEFAULT_MODEL = "wuyan_bpe_v3_sft"
DEFAULT_N = 3

# v3 特殊 token（与 build_bpe_v3.py 顺序一致）
V3_BOS, V3_TITLE, V3_AUTHOR, V3_BODY, V3_EOS, V3_UNK = range(6)


def load_model(model_id="wuyan_bpe"):
    prefix, name, folder, *rest = MODELS[model_id]
    ckpt_name = rest[0] if rest else f"{prefix}model.pt"
    d = ROOT / folder
    ckpt = torch.load(d / ckpt_name, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    itos = json.loads((d / f"{prefix}vocab.json").read_text(encoding="utf-8-sig"))
    itos = {int(k): v for k, v in itos.items()}
    stoi = {c: i for i, c in itos.items()}

    if cfg.get("arch") == "v3":
        from train_v3 import MiniGPT_v3
        model = MiniGPT_v3(vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
                           n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                           dropout=cfg.get("dropout", 0.0))
    else:
        from train import MiniGPT
        model = MiniGPT(vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
                        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                        dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, stoi, itos, cfg, name


@torch.no_grad()
def sample_next(model, stoi, itos, cfg, tokens, temperature, top_k=60,
                banned=None, penalize=None, rep_penalty=0.0):
    """输入 token 序列，采样下一个 token id
    banned: 禁选 token 集合（句内已用字 → 硬性不重复）
    penalize + rep_penalty: 对整首诗已用字对应 token 降权（软惩罚）
    """
    x = torch.tensor(tokens[-cfg["block_size"]:], dtype=torch.long).unsqueeze(0)
    logits, _ = model(x)
    logits = logits[0, -1, :] / max(temperature, 1e-6)
    # 顺序：先禁重/降权，再 top-k 截断（否则候选可能全被禁导致 softmax 出错）
    if banned:
        logits[list(banned)] = -float("Inf")
    if rep_penalty > 0 and penalize:
        for t in penalize:
            logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = -float("Inf")
    # 保险：若全部候选被禁（理论极少发生），退回不禁止重
    if not torch.isfinite(logits).any():
        logits = logits * 0  # 全 0 → 均匀采样
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


# 句边界标点：遇到这些就认为"一句结束"（仅用于句内重复检测）
_LINE_END = set("，。！？；、\n")
_PUNCT_SPLIT = re.compile(r"[，。！？；、\n]")


def _seg_dup_free(text):
    """整首诗块内，每个以标点分隔的片段都不重复字（句内无重复）"""
    for seg in _PUNCT_SPLIT.split(text):
        seg = seg.strip()
        if len(seg) >= 2 and len(seg) != len(set(seg)):
            return False
    return True


from rhyme_dict import evaluate_poem, _is_author_line


def _decode_v3(tokens, itos):
    """把 v3 特殊 token 流解码为《题》/作者/诗句 文本块（与 v2 展示格式一致）"""
    s = ""
    for t in tokens:
        ch = itos[t]
        if ch == "<|eos|>":
            break
        if ch == "<|bos|>":
            continue
        if ch == "<|title|>" or ch == "<|author|>":
            s += "\n"
        elif ch == "<|body|>":
            s += ""
        elif ch == "<|unk|>":
            s += "□"
        else:
            s += ch
    return s.strip()


def _sample_block(model, stoi, itos, cfg, seed, temperature, is_v3):
    """自回归采样一个完整诗块；v3 用特殊 token 引导，遇 <|eos|> 自动停机"""
    max_tokens = 60 + 80
    if is_v3:
        prefix = [stoi.get("<|bos|>", 0)]
        prefix += [stoi.get(c, stoi.get("<|unk|>", 0)) for c in f"《{seed}》"]
        prefix += [stoi.get("<|title|>", 1)]
    else:
        prefix = [stoi[c] for c in seed if c in stoi] or [stoi.get("《", 0)]
    tokens = list(prefix)
    for _ in range(max_tokens):
        nxt = sample_next(model, stoi, itos, cfg, tokens, temperature)
        tokens.append(nxt)
        if is_v3 and itos.get(nxt) == "<|eos|>":
            break
    if is_v3:
        return _decode_v3(tokens, itos)
    return "".join(itos[t] for t in tokens)


def gen_poems(model, stoi, itos, cfg, seed, temperature, n, max_rounds=3):
    """自回归生成候选诗，并通过平水韵与五言格律评分器进行 Best-of-N 优选重排
    1. 底层采样 K = n * 4 候选（单次发散，多轮补齐）
    2. 符号化格律评分 (evaluate_poem)：五言字数、二四句押韵、句内重字、高频字泛滥
    3. 按总得分降序挑选最高品质的前 n 首返回，作者行附韵部印记（如〔押七尤〕）
    """
    is_v3 = cfg.get("arch") == "v3"
    candidates = []
    target_k = max(n * 4, 12)
    for _ in range(max_rounds):
        text = _sample_block(model, stoi, itos, cfg, seed, temperature, is_v3)
        if is_v3:
            block = text
            if block.split("\n")[0].startswith("《"):
                candidates.append(block)
        else:
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            candidates += [b for b in blocks if b.split("\n")[0].startswith("《")]
        if len(candidates) >= target_k:
            break

    if not candidates:
        return []

    # 去除完全相同候选
    seen = set()
    unique_candidates = []
    for b in candidates:
        if b not in seen:
            seen.add(b)
            unique_candidates.append(b)

    # 符号化格律评估与 Best-of-N 优选打分
    scored = []
    for b in unique_candidates:
        ev = evaluate_poem(b, meter=5)
        scored.append((ev["score"], ev, b))

    # 按综合得分从高到低排序
    scored.sort(key=lambda x: x[0], reverse=True)

    # 组装返回文本：作者名统一显示为「佚名」，原预测作者名作为风格标签（如「刘禹锡风格」）；
    # 若成功押韵，在佚名后附韵部小标签（如〔押七尤〕）
    final_poems = []
    for score_val, ev, raw_b in scored[:n]:
        lines = raw_b.split("\n")
        for li in range(1, min(len(lines), 4)):
            if _is_author_line(lines[li]):
                style = lines[li].strip()
                tag = f"  〔押{ev['rhyme_name']}〕" if ev["is_rhymed"] and ev["rhyme_name"] else ""
                lines[li] = f"佚名（{style}风格）{tag}"
                break
        final_poems.append("\n".join(lines))

    return final_poems


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else "春日"
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    n = min(int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N, 6)
    model_id = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_MODEL
    if model_id not in MODELS:
        print(f"未知模型 {model_id}，可用: {', '.join(MODELS)}")
        return

    model, stoi, itos, cfg, name = load_model(model_id)
    print(f"模型已加载（{name}）| 开头「{seed}」| 温度 {temperature} | 出 {n} 首\n")

    poems = gen_poems(model, stoi, itos, cfg, seed, temperature, n)
    if not poems:
        print("（这次生成没有切出完整诗，换个开头或温度再试）")
        return

    out = [f"# 一键写诗 · {name} · 开头「{seed}」· 温度 {temperature}", ""]
    for i, p in enumerate(poems, 1):
        print(f"【第 {i} 首】")
        print(p)
        print()
        out.append(f"【第 {i} 首】")
        out.append(p)
        out.append("")

    prefix, _, folder, *_rest = MODELS[model_id]
    save = ROOT / folder / f"{prefix}poems.txt"
    save.write_text("\n".join(out), encoding="utf-8")
    print(f"已保存: {save.name}")


if __name__ == "__main__":
    main()
