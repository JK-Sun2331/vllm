# SlackKV vLLM 环境

本 fork 保留请求到达、首次获选、完成、KV 受阻归因和跨层 request ID 事件。不能用单独的 `pip install vllm` 替代，否则会丢失本项目的 profiling。

完整迁移入口：同级 `slackkv-control/deployment/MIGRATION.md`。当前环境为 Python 3.12、vLLM 0.28.0+precompiled、torch 2.13.0+cu129；精确已安装依赖记录在 control 的 `deployment/vllm-installed.txt`。

安装：先设置数据盘 `UV_CACHE_DIR`，再执行 `bash tools/codex/install_slackkv.sh`。脚本固定修改前的上游 commit 对应 CUDA 12.9 预编译扩展，Python 部分 editable 安装本 fork；不会重建或替换现有 profiling 代码。该远程 wheel 是否仍可下载、依赖是否可解析，需在新服务器验证，本机未重新安装。

如果指定的 wheel 不可用：不要改用 latest/nightly 掩盖版本差异。保存安装错误，检查该 commit 的官方 wheel 索引；如果确实缺失，在同一 fork 提交上按 `docs/getting_started/installation/gpu.md` 的源码构建方式安装，需要匹配 CUDA toolkit、编译器和 Rust。不要将旧机器 `.venv` 直接复制成新环境。

安装本身不启动推理；导入 torch/vLLM 或运行实验前先 `nvidia-smi`，选择空闲卡并设置 `CUDA_VISIBLE_DEVICES`。使用 migration 文档中的冷门端口，服务只能由本轮进程停止。

gpt-oss 在 Ampere 上的量化后端、Harmony 输出以及混合注意力的 KV profiling/offload 尚未在本项目中实测。基础聊天成功不等于跨层采集验收通过。
