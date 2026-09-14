# -*- coding: utf-8 -*-
"""
训练守护脚本：反复调用 train.py，直到 {prefix}train_state.json 记录的步数达标。
即使子进程被系统回收/崩溃，也会自动重启并从 checkpoint 继续，保证进度不丢。
用法: python keep_training.py [数据前缀] [总步数]
示例: python keep_training.py jueju_ 1000
"""
import json
import subprocess
import sys
import time
from pathlib import Path

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


def current_step(prefix: str) -> int:
    state_path = data_dir(prefix) / f"{prefix}train_state.json"
    if state_path.exists():
        try:
            return int(json.loads(state_path.read_text(encoding="utf-8"))["step"])
        except Exception:
            return 0
    return 0


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    script = sys.argv[3] if len(sys.argv) > 3 else "train.py"

    attempts = 0
    while True:
        done = current_step(prefix)
        if done >= max_steps:
            print(f"[keep] 训练已完成: {done}/{max_steps} 步")
            return
        attempts += 1
        print(f"[keep] 第 {attempts} 次启动 {script} (当前 {done}/{max_steps} 步)")
        proc = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts" / script), prefix, str(max_steps)])
        done = current_step(prefix)
        if done >= max_steps:
            print(f"[keep] 训练完成: {done}/{max_steps} 步")
            return
        if proc.returncode == 0:
            print(f"[keep] 进程正常退出但未到目标步数({done}/{max_steps}), 3 秒后重启")
            time.sleep(3)
        else:
            print(f"[keep] 进程退出码 {proc.returncode}, 5 秒后重启")
            time.sleep(5)


if __name__ == "__main__":
    main()
