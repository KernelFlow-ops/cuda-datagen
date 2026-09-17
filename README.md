# CUDA 算子 SFT 数据生成

用 OpenRouter 上的大模型，为 `question.jsonl` 生成可 `nvcc -c` 编译的 CUDA 算子代码，写出 SFT jsonl。流程用 LangGraph 描述：生成 → 编译 → 失败则修正（每候选最多 3 次）→ 换候选（最多 3 个）→ 仍失败则放弃该题。

判定只看编译是否通过，不跑测试、不比对数值。

## 准备

使用 conda 环境 `langchain`（其中已安装 langgraph；本机没有名为 `langgraph` 的 env）。

```bash
conda activate langchain
# 如缺依赖：pip install -r requirements.txt
cp .env.example .env   # 若还没有 .env
```

在 `.env` 填入对应 provider 的 key。可改的项：

| 变量 | 默认 | 含义 |
|---|---|---|
| `LLM_PROVIDER` | `openrouter` | `openrouter`（Anthropic Messages）或 `nvidia`（官方 OpenAI 兼容接口） |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | OpenRouter | `LLM_PROVIDER=openrouter` 时使用 |
| `NVIDIA_API_KEY` / `_2` / `_3` / `NVIDIA_BASE_URL` | NIM | 最多 3 个 NVIDIA key（也可在 `NVIDIA_API_KEY` 里逗号分隔）；每个 key 独立占一个 nvidia 槽位。默认 `https://integrate.api.nvidia.com/v1` |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter 模型 |
| `NVIDIA_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA 官方模型 |
| `MODEL` | 空 | 非空时覆盖上面两个，一般留空 |
| `THINKING_LEVEL` | `medium` | OpenRouter 走 `reasoning.effort`；NVIDIA 走 `chat_template_kwargs.enable_thinking` |
| `MAX_INPUT_TOKENS` | `131072` | 输入上下文上限（超出则丢掉旧轮、截尾；约按字符估算 token） |
| `MAX_OUTPUT_TOKENS` | `50000` | 输出 token 上限，对应 API 的 `max_tokens` |
| `MAX_TOKENS` | `50000` | 兼容别名；若未设 `MAX_OUTPUT_TOKENS` 则用它 |
| `TOP_P` | `0.95` | NVIDIA Chat Completions 的 top_p |
| `MAX_CANDIDATES` | `3` | 每题最多候选数 |
| `MAX_REPAIRS` | `3` | 每个候选最多修正次数 |
| `WORKERS` | `1` | `WORKERS_PER_PROVIDER=0` 时的总进程数；nvidia worker 在多个 key 间轮询 |
| `JUDGE_ENABLED` | `true` | 启用 Judge Agent 代码质量评估 |
| `USE_JUDGE_OPTIMIZATION` | `false` | 是否应用 Judge 的轻量优化建议（需重新编译验证） |
| `COT_ENABLED` | `true` | 采集 API reasoning/thinking，并走 CoT 节点 |
| `COT_AGENT_ENABLED` | `true` | 用 CoT Agent 把原始 thinking 整理成教学型 CoT（额外一次 LLM 调用） |
| `COT_IN_ASSISTANT` | `true` | 训练标签是否包 `<think>…</think>` + 源码 |
| `COT_TEMPERATURE` | `0.2` | CoT Agent 采样温度 |
| `COT_MAX_CHARS` | `8000` | 整理后 CoT 字符上限 |
| `COT_RAW_MAX_CHARS` | `24000` | 送进 Agent 的原始 thinking 上限 |
| `COT_RAW_STORE_MAX_CHARS` | `32768` | `sft.jsonl` metadata 里归档 raw thinking 的上限 |
| `COT_ON_EMPTY` | `synthetic` | 教师 thinking 为空时：`synthetic`（由终稿代码反写）/ `empty` |
| `COT_ON_AGENT_FAIL` | `raw` | Agent 失败回退：`raw` / `synthetic` / `empty` |
| `KERNEL_DIALECTS` | `cuda` | 逗号分隔：`cuda`,`cutlass`（CUTLASS **4.x** + CuTe，别名 `cute`）,`triton`,`tilelang` |
| `KERNEL_MODE` | `single` | `single` 只跑一种；`all` 每题把列出的方言各生成一遍 |
| `KERNEL_DIALECT` | 空 | `single` 时覆盖列表第一项 |
| `CUTLASS_HOME` | `/usr/local/cutlass-4.3.5` | CUTLASS 4.x 根目录；`available()` 要求 `CUTLASS_MAJOR==4` |
| `CUTLASS_CXX_STD` | `c++17` | cutlass 方言 `nvcc -std=` |
| `TRITON_TIMEOUT_SEC` | `90` | Triton import/JIT 门闩超时 |
| `TILELANG_TIMEOUT_SEC` | `180` | TileLang 门闩超时；未安装则自动跳过 |
| `ASYNC_LLM_ENABLED` | `true` | 启用异步 LLM 调用（编译期间预生成修复轮响应） |
| `ASYNC_LLM_MAX_WORKERS` | `2` | 异步 LLM 并发数（1-4） |
| `LLM_PROVIDERS` | 空 | 逗号分隔多 provider，如 `nvidia,openrouter` |
| `WORKERS_PER_PROVIDER` | `0` | 每个槽位的并发数；NVIDIA 每个 key 各算一个槽位。例如 3 个 NVIDIA key + openrouter、值为 2 → 6 个 nvidia + 2 个 openrouter |
| `REPAIR_ERROR_MAX_CHARS` | `6000` | 修复轮 nvcc 日志上限；超过则去重摘要，不超过则全文 |
| `WORK_KEEP` | `simple` | `simple`：每题 `work/q{id}/` 只留最后一份 `solution.cu`；`detailed`：保留全部 `c*/r*` 尝试 |
| `CUDA_ARCH` | 空则自动探测 | 如 `sm_86` |
| `GPU_NAME` | 空则自动探测 | 写入 prompt |

NVIDIA 官方示例对应配置：

```
LLM_PROVIDER=nvidia
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=nvapi-...
NVIDIA_API_KEY_2=nvapi-...
NVIDIA_API_KEY_3=nvapi-...
MODEL=nvidia/nemotron-3-ultra-550b-a55b
THINKING_LEVEL=medium
```

三个 NVIDIA key 用来提高 NIM 并发（额度按 key 独立计算）。配合 `LLM_PROVIDERS=nvidia,openrouter` 和 `WORKERS_PER_PROVIDER=1` 时，会启动 3 个 nvidia worker（每 key 一个）+ 1 个 openrouter worker。日志里只打印 `nvidia#1` / `nvidia#2` / `nvidia#3`，不会写出 key 本身。

本机默认会探测到 RTX 3060 / `sm_86` / CUDA 12.6。

## 运行

先确认本机 nvcc 可用：

```bash
conda activate langchain
python run.py --dry-compile
```

试跑 3 题：

```bash
python run.py --limit 3
```

全量（自动跳过 `data/progress.jsonl` 里已成功或已放弃的题）：

```bash
python run.py
```

也可不手动 activate，直接：

```bash
conda run -n langchain python run.py --limit 3
```

多进程（`.env` 里 `WORKERS`，或命令行覆盖）。jsonl 写入带文件锁；worker>1 时关闭 token 流式打印：

```bash
python run.py --workers 6 --quiet
python run.py --workers 6 --limit 12 --quiet   # 并发试跑
```

常用参数：`--offset N`、`--ids 1,2,10`、`--overwrite`、`--quiet`、`--workers N`、`--data-dir DIR`、`--dialects cuda,cutlass,triton`、`--kernel-mode all`。

只写 CUTLASS 4.x / CuTe：

```bash
python run.py --dialects cutlass --limit 3
```

每题同时写 CUDA + CUTLASS 4 + Triton（TileLang 未安装会自动跳过）：

```bash
python run.py --dialects cuda,cutlass,triton --kernel-mode all --limit 3
```

## 输出

| 文件 | 内容 |
|---|---|
| `data/sft.jsonl` | 生成归档（`messages` + `id`/`metadata`；`metadata.raw_reasoning` 为原始 thinking，`metadata.cot` 为整理结果） |
| `data/sft_ms_swift.jsonl` | **ms-swift SFT** 标准 `messages` 格式 |
| `data/sft_openrlhf.jsonl` | **OpenRLHF SFT** 的 `input`/`output` 对话格式 |
| `data/abandoned.jsonl` | 3 个候选都失败的题目和最后编译错误 |
| `data/progress.jsonl` | 断点续跑 |
| `data/run.log` | 当次运行日志（每次启动会清空旧的 `run*.log`，多进程共用这一份） |
| `work/` | `WORK_KEEP=simple` 时每题只留最后 `solution.cu`；`detailed` 时为 `q{id}/c{c}/r{r}/` |

默认 SFT assistant 是「整理后的 CoT + 抽取后的 CUDA 源码」：

```
<think>
1. Problem restatement
...
</think>
#include <cuda_runtime.h>
...
```

原始 API thinking 写在 `sft.jsonl` 的 `metadata.raw_reasoning`，不进 ms-swift / OpenRLHF。`COT_IN_ASSISTANT=false` 时训练 jsonl 仍为纯代码。已有 `sft.jsonl` 时可以只做格式导出（不调 API）：

```bash
python run.py --export-sft
```

### ms-swift

每行只有 `messages`（可选 system + user + assistant）：

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

```bash
swift sft \
  --model <your-model> \
  --dataset data/sft_ms_swift.jsonl \
  --tuner_type lora \
  --output_dir output/cuda-sft-swift
```

### OpenRLHF

`input` 是 prompt 侧 messages（system+user），`output` 是 assistant 目标：

```json
{"input": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "output": [{"role": "assistant", "content": "..."}]}
```

```bash
deepspeed --module openrlhf.cli.train_sft \
  --dataset data/sft_openrlhf.jsonl \
  --input_key input \
  --output_key output \
  --apply_chat_template \
  --pretrain <your-model> \
  --save_path ./checkpoint/cuda-sft-openrlhf
```

## 图结构

```
prepare → generate → extract → compile
                         ├ compile ok → judge → cot → save_success
                         ├ repair < 3 → repair → generate
                         ├ candidate < 3 → next_candidate → generate
                         └ else → save_abandoned
```

`KERNEL_MODE=all` 时在 CLI 层按方言展开 job（每题每种语言各跑一张上图），进度键是 `(id, dialect)`。CUTLASS 方言钉 **CUTLASS 4.x + CuTe**（本机 `/usr/local/cutlass-4.3.5`）。

LLM：`LLM_PROVIDER=openrouter` 时走 Anthropic Messages（`POST {OPENROUTER_BASE_URL}/v1/messages`）；`nvidia` 时走 OpenAI Chat Completions 流式（`{NVIDIA_BASE_URL}/chat/completions`）。生成时采集 reasoning/thinking（OpenRouter：`thinking` block / `reasoning` 字段；NVIDIA：`delta.reasoning_content`，必要时再从 `<think>` 标签兜底）。`judge` 与 `cot` 在开关关闭时 no-op。CoT Agent 只整理胜出样本的推理，不改已经通过编译的代码。高质量 CoT 建议保持 `THINKING_LEVEL=medium` 或 `high`，并让 `MAX_OUTPUT_TOKENS` 明显大于思考预算。
