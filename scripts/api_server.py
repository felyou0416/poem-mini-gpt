# -*- coding: utf-8 -*-
"""
写诗模型 · HTTP API 服务（支持 3 个模型切换）
启动: python scripts\api_server.py [端口]      # 默认 127.0.0.1:5000

调用方式（温度默认 0.6、模型默认 wuyan_bpe，可不传）:
  POST /api/poem   body={"seed":"秋思","temperature":0.8,"n":3,"model":"jueju"}
  GET  /api/poem?seed=秋思&temperature=0.6&n=3&model=jueju
  GET  /api/models  列出可用模型
  GET  /health      健康检查

可用模型: wuyan_bpe（五言BPE v2，当前最强）/ jueju（五言绝句）/ all（通用33万首）
返回: {"model":"wuyan_bpe","model_name":"五言BPE v2","seed":"...","temperature":0.6,"n":3,"poems":[...]}
"""
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from poem import load_model, gen_poems, MODELS

ROOT = Path(__file__).resolve().parent.parent

app = Flask(__name__)
CACHE = {}  # model_id -> (model, stoi, itos, cfg, name)，进程内只加载一次

# 前端下拉框用的展示信息（顺序即展示顺序）
MODEL_INFO = [
    {"id": "wuyan_bpe_v3_sft", "name": "五言BPE v3·SFT", "desc": "9.2万首预训练 + 3千首唐风名篇微调 · 当前最强"},
    {"id": "wuyan_bpe_v3",    "name": "五言BPE v3",     "desc": "9.2万首严格五言 · RoPE+RMSNorm 6层 · 全新架构"},
    {"id": "wuyan_bpe",       "name": "五言BPE v2",     "desc": "11万首五言 · BPE分词 · 上一代"},
    {"id": "jueju",           "name": "五言绝句",       "desc": "1.7万首绝句 · 格式最工整"},
    {"id": "all",             "name": "通用模型",       "desc": "33万首全语料 · 五言七言杂言"},
]


@app.after_request
def add_cors(resp):
    """允许跨域，方便任何前端/脚本调用"""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/")
def index():
    return send_file(ROOT / "前端" / "ui.html")


@app.route("/api/models")
def api_models():
    return jsonify({"models": MODEL_INFO})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "loaded": sorted(CACHE.keys())})


def get_model(model_id):
    """按 id 取模型（缓存复用）"""
    if model_id not in MODELS:
        raise ValueError(f"未知模型 {model_id}，可用: {', '.join(MODELS)}")
    if model_id not in CACHE:
        CACHE[model_id] = load_model(model_id)
        print(f"[api] 模型已加载: {MODELS[model_id][1]}")
    return CACHE[model_id]


@app.route("/api/poem", methods=["GET", "POST"])
def api_poem():
    # 解析参数：POST JSON 优先，否则 GET query；缺省用默认值
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        seed = str(data.get("seed", "秋思"))[:20]
        temperature = float(data.get("temperature", 0.6))
        n = int(data.get("n", 3))
        model_id = str(data.get("model", "wuyan_bpe_v3_sft"))
    else:
        seed = str(request.args.get("seed", "秋思"))[:20]
        temperature = float(request.args.get("temperature", 0.6))
        n = int(request.args.get("n", 3))
        model_id = str(request.args.get("model", "wuyan_bpe_v3_sft"))

    # 参数保护：温度 0.1~2.0，数量 1~6
    temperature = max(0.1, min(temperature, 2.0))
    n = max(1, min(n, 6))

    try:
        model, stoi, itos, cfg, name = get_model(model_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    poems = gen_poems(model, stoi, itos, cfg, seed, temperature, n)

    return jsonify({
        "model": model_id,
        "model_name": name,
        "seed": seed,
        "temperature": temperature,
        "n": len(poems),
        "poems": poems,
    })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"[api] 写诗服务已启动: http://127.0.0.1:{port}/")
    print(f"[api] 模型: {', '.join(m['name'] for m in MODEL_INFO)} | 默认温度 0.6")
    app.run(host="127.0.0.1", port=port)
