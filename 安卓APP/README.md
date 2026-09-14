# 写诗 AI · 安卓端离线推理客户端

本项目为 **写诗 AI（MiniGPT_v3·SFT）** 的原生 Android 客户端。采用纯 Kotlin 实现轻量级 Transformer 推理引擎与平水韵格律评分体系，无需网络连接、无需第三方重型推理框架，在手机端即可进行纯本地、毫秒级的五言古诗创作。

---

## 🎯 核心架构与模型策略

### 1. 单一模型轻量化设计（Strictly Single Model）
- **打包模型**: 仅内置单一最佳模型——**五言BPE v3·SFT**（唐风名篇微调旗舰版）。
- **模型文件**: `app/src/main/assets/model.bin`（约 17.3MB，344 万参数，6 层 / 192 维 / 6 头 RoPE + RMSNorm）。
- **杜绝冗余**: 严禁将其他基座模型、绝句模型或 PyTorch 原始 `.pt` 大文件打包入内，确保整体 APK 安装包控制在 **~20MB** 左右，手机冷启动极速且省内存。

### 2. 算子与推理实现
- **自回归推理**: 完整复刻 `KV-Cache` 增量注意力机制（单步只推算新 token）。
- **格律优选**: 内置平水韵十四韵评分器（7,183 字字符倒排索引），通过 Best-of-N 多候选采样与格律打分，自动选拔押韵、意境俱佳的名作，并在作者行附韵部印记（如 `〔押十一尤〕`）。
- **现代架构**: RoPE 旋转位置编码、精确 erf 近似 GELU、RMSNorm 与权重共享（lm_head / tok_emb）。
- **GC 极致优化**: 零对象装箱的快速 Top-K 多项式采样，避免 Android 垃圾回收卡顿。

---

## ☁️ GitHub 云端自动构建 (GitHub Actions CI)

本项目已配置标准 GitHub Actions 自动化构建工作流 [`.github/workflows/build-apk.yml`](../.github/workflows/build-apk.yml)。

### 1. 触发方式
- **代码提交 (Push)**: 推送代码至 `main`/`master` 分支且修改了 `安卓APP/**` 目录时自动触发构建。
- **手动一键触发 (workflow_dispatch)**: 
  1. 进入 GitHub 仓库页面，点击顶部 **Actions** 标签页；
  2. 在左侧选择 **Android CI - Build Poetry AI APK** 工作流；
  3. 点击右侧 **Run workflow** 按钮即可在云端即时构建。
- **Release 发布 (Tag)**: 推送以 `v` 开头的标签（如 `git tag v1.0.0 && git push origin v1.0.0`）会自动构建并将 APK 挂载到 GitHub Releases 附件。

### 2. 下载构建完成的 APK
1. 进入对应构建运行的详情页；
2. 页面底部 **Artifacts** 区域即可点击下载 `写诗AI-v3-SFT-安卓安装包`（解压后即为签名的 `写诗AI_v3_sft.apk`）。
3. 安装到 Android 手机直接运行即可。

---

## 💻 本地构建指引

若在本地环境开发，可在 Android Studio 中直接打开本目录 `安卓APP/`：
- **JDK**: Java 17 (Temurin / Microsoft JDK)
- **AGP**: 8.4.2 / Gradle 8.7
- **构建命令**:
  ```bash
  gradle assembleDebug
  ```
  构建产物位于 `app/build/outputs/apk/debug/app-debug.apk`。
