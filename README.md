# IndexTTS HTTP 服务

IndexTTS HTTP Service 提供基于 IndexTTS-2 的文本转语音（TTS）能力，支持在保持说话人音色的前提下进行情感可控的语音合成。项目提供 Docker 化部署、OpenAPI 定义与测试脚本，开箱即用。

## 特性

- 说话人音色克隆：通过参考音频 `prompt_audio` 克隆音色
- 多种情感控制模式：
  - 说话人感情复制：复制参考音频 `prompt_audio` 感情
  - 参考音频情感混合（reference）：`emotion_audio` + `emotion_weight`
  - 八维情感因子（vector）：`happy/angry/sad/afraid/disgusted/melancholic/surprised/calm`
  - 情感文本（text）：`emotion_text`
- 简洁的 HTTP API（FastAPI 实现），配套 `openapi.yml`
- 提供 Dockerfile / docker-compose / Makefile，便于部署与运行

## 目录结构

- `main.py`：HTTP 服务入口（FastAPI）
- `openapi.yml`：OpenAPI 3.0.3 定义
- `Dockerfile`、`docker-compose.yml`：容器化部署配置
- `Makefile`：构建、运行、下载模型、OpenAPI Lint 等命令
- `examples/`：示例音频
- `test.py`：四种场景的端到端测试脚本
- `checkpoints/`：模型权重目录（运行时挂载/下载）

## 快速开始

### 方法一：使用 Makefile（推荐）

```bash
# 下载模型到本地 checkpoints/（首次运行需要）
make download

# 启动服务（默认监听 9010）
make run

# 查看日志
make logs

# 停止服务
make stop
```

服务启动后，默认地址为 `http://localhost:9010`。

### 方法二：docker-compose（使用 GHCR 镜像）

```bash
# 首次下载模型权重（到本地目录）
make download

# 启动
docker compose up -d
```

- 端口映射：`9010:9010`
- 目录挂载：
  - `./huggingface:/root/.cache/huggingface`
  - `./checkpoints:/app/checkpoints`
  - `./prompt_cache:/app/prompt_cache`
- 环境变量（可选）：
  - `HF_ENDPOINT=https://hf-mirror.com`
  - `USE_FP16=0|1`
  - `USE_CUDA_KERNEL=0|1`

如使用 NVIDIA GPU，可参考 `docker-compose.yml` 注释启用 `deploy.resources.devices`，并将 `USE_FP16/USE_CUDA_KERNEL` 设置为 `1`（需正确安装驱动及满足 Docker 版本要求）。

## 环境变量

- `HF_ENDPOINT`：HuggingFace 镜像地址（默认 `https://hf-mirror.com`）
- `USE_FP16`：是否使用 FP16 推理（`0`/`1`）
- `USE_CUDA_KERNEL`：是否启用 CUDA 内核（`0`/`1`）
- `PROMPT_CACHE_DIR`：参考音频缓存目录（默认在系统临时目录下的 `indextts/prompts`）

## 模型权重

首次运行需要下载 IndexTTS-2 模型至 `checkpoints/`：

```bash
make download
# 等价于在 python:3.10-slim 容器内执行：
# pip install modelscope && modelscope download --model IndexTeam/IndexTTS-2 --local_dir /local/checkpoints
```

下载完成后，`checkpoints/` 下应存在 `config.yaml` 等文件。

## API 概览

服务提供 4 个主要端点。成功返回值均为 Base64 编码的 WAV 音频（JSON 字符串）。错误返回统一为：

```json
{
  "code": "BAD_REQUEST|BUSY|INTERNAL_ERROR|HTTP_ERROR",
  "message": "..."
}
```

- `POST /synthesize/speaker`：使用说话人音色和感情
- `POST /synthesize/reference`：使用说话人音色 + 参考情感音频
- `POST /synthesize/vector`：使用八维情感因子控制情感
- `POST /synthesize/text`：使用情感文本控制情感

详细字段、约束、示例请查看仓库内 `openapi.yml`（可用 `make lint` 进行文档校验）。

### 公共字段（Base）

- `prompt_audio`（string, byte）：Base64-encoded `.wav` 说话人参考音频
- `text`（string）：待合成文本
- `max_text_tokens_per_segment`（integer, default=120）：每段文本最大 token 数

### 生成参数 `generation_args`

- `do_sample`（bool，默认 true）：开启随机采样；设为 false 则主要由 beam search 决定输出。
- `top_p`（float，0–1，默认 0.8）：核采样阈值，仅在 `do_sample=true` 时生效；常用 0.7–0.95。
- `top_k`（int，≥0，默认 30）：Top-K 采样；`0` 表示禁用；常用 20–100。
- `temperature`（float，>0，默认 0.8）：采样温度；越大越随机，常用 0.6–1.2。
- `length_penalty`（float，默认 0.0）：beam 搜索长度惩罚；>1 倾向更长，<1 倾向更短。
- `num_beams`（int，≥1，默认 3）：beam 数；>1 启用 beam search（可与采样组合，具体以实现为准）。
- `repetition_penalty`（float，≥1，默认 10.0）：重复惩罚；越大越抑制重复。建议 1.0–5.0 按文本调整（过大可能导致断续）。
- `max_mel_tokens`（int，默认 1500）：最大梅尔 token 上限；过小会截断音频，增大可生成更长音频但更耗时/显存。

提示：对于较长文本，建议配合合理的 `max_text_tokens_per_segment` 进行分段合成以提升稳定性。

### 各模式额外字段

- reference（参考情感混合）：
  - `emotion_audio`（string, byte，必需）：Base64-encoded `.wav` 情感参考音频。
  - `emotion_weight`（float，0–1，默认 0.8）：情感混合权重；服务内部会再乘以 0.8 进行缩放，避免过强失真。
  - 该模式不使用情感向量；说话人音色来自 `prompt_audio`，情感来自 `emotion_audio`。

- vector（情感因子向量）：
  - `emotion_factors`（object，必需）：包含八个 0–1 浮点值字段，顺序与含义如下：
    - `happy`、`angry`、`sad`、`afraid`、`disgusted`、`melancholic`、`surprised`、`calm`
  - 服务内部会对该八维向量进行加权与归一化，使用权重系数近似为 `[0.75, 0.70, 0.80, 0.80, 0.75, 0.75, 0.55, 0.45]` 并限制总强度不超过约 `0.8`，以防过饱和导致失真。
  - `emotion_random`（bool，默认 false）：启用随机情感采样（可能降低音色克隆保真度）。
  - 该模式忽略 `emotion_audio`，且 `emo_alpha` 不参与混合（以情感向量为准）。

- text（情感文本）：
  - `emotion_text`（string，可为 null）：情感描述文本；允许为 `null`，但不允许为空字符串（空字符串会被拒绝）。
  - `emotion_random`（bool，默认 false）：启用随机情感采样（可能降低音色克隆保真度）。
  - 该模式从文本推导情感，不使用外部情感音频或情感向量。

### cURL 示例

以下示例仅为结构示意，实际调用请替换 `<BASE64_WAV>` 与文本：

```bash
# 1) speaker
curl -s -X POST http://localhost:9010/synthesize/speaker \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_audio": "<BASE64_WAV>",
    "text": "Translate for me, what is a surprise!",
    "max_text_tokens_per_segment": 120,
    "generation_args": {
      "do_sample": true,
      "top_p": 0.8,
      "top_k": 30,
      "temperature": 0.8,
      "length_penalty": 0.0,
      "num_beams": 3,
      "repetition_penalty": 10.0,
      "max_mel_tokens": 1500
    }
  }'

# 2) reference
curl -s -X POST http://localhost:9010/synthesize/reference \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_audio": "<BASE64_WAV>",
    "emotion_audio": "<BASE64_WAV>",
    "emotion_weight": 1.0,
    "text": "你看看你，对我还有没有一点父子之间的信任了。",
    "max_text_tokens_per_segment": 120,
    "generation_args": { "do_sample": true, "top_p": 0.8, "top_k": 30, "temperature": 0.8, "length_penalty": 0.0, "num_beams": 3, "repetition_penalty": 10.0, "max_mel_tokens": 1500 }
  }'

# 3) vector（惊讶=1.0）
curl -s -X POST http://localhost:9010/synthesize/vector \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_audio": "<BASE64_WAV>",
    "text": "哇塞！这个爆率也太高了！欧皇附体了！",
    "max_text_tokens_per_segment": 120,
    "emotion_factors": { "happy":0, "angry":0, "sad":0, "afraid":0, "disgusted":0, "melancholic":0, "surprised":1.0, "calm":0 },
    "emotion_random": false,
    "generation_args": { "do_sample": true, "top_p": 0.8, "top_k": 30, "temperature": 0.8, "length_penalty": 0.0, "num_beams": 3, "repetition_penalty": 10.0, "max_mel_tokens": 1500 }
  }'

# 4) text（情感文本）
curl -s -X POST http://localhost:9010/synthesize/text \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_audio": "<BASE64_WAV>",
    "emotion_text": "极度悲伤",
    "text": "这些年的时光终究是错付了... ",
    "max_text_tokens_per_segment": 120,
    "emotion_random": false,
    "generation_args": { "do_sample": true, "top_p": 0.8, "top_k": 30, "temperature": 0.8, "length_penalty": 0.0, "num_beams": 3, "repetition_penalty": 10.0, "max_mel_tokens": 1500 }
  }'
```

## 测试与示例脚本

确保服务已运行，并存在 `examples/` 示例音频：

```bash
python test.py --examples-dir examples --outputs-dir outputs
```

脚本会顺序调用四个端点，并将生成结果保存至 `outputs/`。脚本带有 429 重试逻辑（服务内部采用互斥锁串行推理，忙时返回 429）。

## 常见问题

- 返回 429：表示服务正忙，请稍后重试（或串行化调用）。
- 参考音频格式：需为 `.wav`，以 Base64 传输。
- 并发：默认单实例串行处理，避免显存/内存不足。
- GPU：如需使用 GPU，请正确配置 NVIDIA 驱动、Docker，并设置 `USE_FP16=1`、`USE_CUDA_KERNEL=1`。

## 许可证

本项目基于 MIT 许可证开源，详见 `LICENSE`。

## 致谢

- IndexTTS 项目（https://github.com/index-tts/index-tts）
- PyTorch / FastAPI / Uvicorn
