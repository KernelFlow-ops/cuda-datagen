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

在 `.env` 填入 `OPENROUTER_API_KEY`。可改的项：

| 变量 | 默认 | 含义 |
|---|---|---|
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api` | Anthropic Messages 的 base（SDK 会请求 `/v1/messages`） |
| `MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter 模型名 |
| `THINKING_LEVEL` | `medium` | 思考等级，经 `reasoning.effort` 发送 |
| `MAX_CANDIDATES` | `3` | 每题最多候选数 |
| `MAX_REPAIRS` | `3` | 每个候选最多修正次数 |
| `CUDA_ARCH` | 空则自动探测 | 如 `sm_86` |
| `GPU_NAME` | 空则自动探测 | 写入 prompt |

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

常用参数：`--offset N`、`--ids 1,2,10`、`--overwrite`、`--quiet`、`--data-dir DIR`。

## 输出

| 文件 | 内容 |
|---|---|
| `data/sft.jsonl` | 生成归档（`messages` + `id`/`metadata`，便于排查） |
| `data/sft_ms_swift.jsonl` | **ms-swift SFT** 标准 `messages` 格式 |
| `data/sft_openrlhf.jsonl` | **OpenRLHF SFT** 的 `input`/`output` 对话格式 |
| `data/abandoned.jsonl` | 3 个候选都失败的题目和最后编译错误 |
| `data/progress.jsonl` | 断点续跑 |
| `work/q{id}/c{candidate}/r{repair}/` | 每次尝试的 `solution.cu` 与 stub 头文件 |

SFT 的 assistant 内容是抽取后的 CUDA 源码，不含 thinking。已有 `sft.jsonl` 时可以只做格式导出（不调 API）：

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
                         ├ compile ok → save_success
                         ├ repair < 3 → repair → generate
                         ├ candidate < 3 → next_candidate → generate
                         └ else → save_abandoned
```

LLM 走 Anthropic Messages 流式接口：`POST {OPENROUTER_BASE_URL}/v1/messages`。
