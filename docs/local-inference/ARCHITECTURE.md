# MERLIN Local Inference Architecture

> **Status:** Proposal / RFC
> **Branch:** `feature/local-inference-architecture`
> **Date:** 2026-03-22

## Executive Summary

This document defines the architecture for running MERLIN entirely on local infrastructure, replacing all cloud API dependencies (Anthropic Claude, ElevenLabs TTS) with fine-tuned open-source models. The result is a self-contained, air-gappable aviation AI co-pilot with zero runtime cloud dependencies.

---

## Current vs. Local Architecture

### Cloud Dependencies Being Replaced

| Current Service | Cloud Provider | Replacement | Model |
|---|---|---|---|
| LLM Inference | Anthropic Claude API | Local vLLM/SGLang | Qwen3.5-35B-A3B (fine-tuned) |
| Text-to-Speech | ElevenLabs API | Local Kokoro / F5-TTS | Kokoro v0.19+ (fine-tuned voice) |
| Speech-to-Text | Self-hosted faster-whisper | Self-hosted faster-whisper | Whisper large-v3-turbo (fine-tuned) |
| Embeddings | ChromaDB default | Local sentence-transformer | aviation-tuned bge-base-en-v1.5 |
| Airport Lookup | aviationapi.com REST | Local SQLite + OurAirports | FAA NASR data dump |

### Services Already Local (No Change)

| Service | Technology | Notes |
|---|---|---|
| SimConnect Bridge | C# .NET 8, WebSocket | Runs on Windows host |
| Vector Store | ChromaDB | Already Dockerized |
| Web UI | FastAPI + static frontend | Already local |
| Audio Preprocessing | Python (scipy, numpy) | Already local |
| VAD | Silero VAD | Already local |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MERLIN LOCAL INFERENCE STACK                      │
│                                                                         │
│  ┌──────────────┐     ┌──────────────────────────────────────────────┐  │
│  │  MSFS 2024   │     │              GPU SERVER                      │  │
│  │  (Windows)   │     │                                              │  │
│  │              │     │  ┌────────────────────────────────────────┐  │  │
│  │  SimConnect  │◄───►│  │  LLM Inference Engine (vLLM / SGLang) │  │  │
│  │  Bridge.exe  │ WS  │  │  Qwen3.5-35B-A3B-aviation (4-bit)    │  │  │
│  │              │     │  │  OpenAI-compatible API [:8081]         │  │  │
│  └──────┬───────┘     │  │  + Outlines constrained decoding      │  │  │
│         │             │  └────────────────────────────────────────┘  │  │
│         │ WS :8080    │                                              │  │
│         ▼             │  ┌────────────────────────────────────────┐  │  │
│  ┌──────────────┐     │  │  STT Engine (faster-whisper)           │  │  │
│  │ Orchestrator │◄───►│  │  whisper-large-v3-turbo-aviation      │  │  │
│  │   (Python)   │ HTTP│  │  OpenAI-compatible API [:9090]         │  │  │
│  │              │     │  └────────────────────────────────────────┘  │  │
│  │  - claude_   │     │                                              │  │
│  │    client.py │     │  ┌────────────────────────────────────────┐  │  │
│  │  - voice.py  │◄───►│  │  TTS Engine (Kokoro / F5-TTS)         │  │  │
│  │  - tools.py  │ HTTP│  │  merlin-voice-v1 (fine-tuned)         │  │  │
│  │  - sim_      │     │  │  Streaming API [:9091]                 │  │  │
│  │    client.py │     │  └────────────────────────────────────────┘  │  │
│  │              │     │                                              │  │
│  └──────┬───────┘     │  ┌────────────────────────────────────────┐  │  │
│         │             │  │  Embedding Engine                      │  │  │
│         │ WS :3838    │  │  bge-base-en-v1.5-aviation             │  │  │
│  ┌──────▼───────┐     │  │  (runs inside ChromaDB or standalone) │  │  │
│  │   Web UI     │     │  └────────────────────────────────────────┘  │  │
│  │  (FastAPI)   │     │                                              │  │
│  │  Browser     │     │  ┌────────────────────────────────────────┐  │  │
│  │  client      │     │  │  ChromaDB [:8000]                      │  │  │
│  └──────────────┘     │  │  + Local airport SQLite [:8082]        │  │  │
│                       │  └────────────────────────────────────────┘  │  │
│                       └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. LLM Inference Engine

**Primary Model: Qwen3.5-35B-A3B (MoE)**

| Property | Value |
|---|---|
| Total Parameters | 35B |
| Active Parameters | 3B per token (256 experts, 8 routed + 1 shared) |
| Quantization | Q4_K_M (GGUF) or FP8 (vLLM) |
| VRAM Required | ~18-22 GB (4-bit) |
| Expected Throughput | 60-100+ tok/s on RTX 5090 (only 3B active) |
| Context Window | 128K tokens |
| Tool Calling | Native, with `qwen3_coder` parser |

**Why Qwen3.5-35B-A3B:**
- MoE architecture means only 3B parameters activate per token, giving near-8B inference speed with 35B-class reasoning quality
- Native tool calling support with dedicated parser in vLLM/SGLang
- Benchmark scores: MMLU-Pro 85.3, GPQA Diamond 84.2, SWE-bench 69.2
- Thinking/non-thinking modes map to MERLIN's dynamic token budgeting (terse acknowledgments vs. detailed briefings)
- 4-bit quantization fits on a single RTX 4090/5090 with room for other models

**Fallback Model: Qwen3.5-9B** (for constrained hardware or faster responses)

**Serving Stack:**
- **Development:** Ollama (`ollama pull qwen3.5:35b-a3b`)
- **Production:** vLLM with Outlines constrained decoding
  ```bash
  vllm serve Qwen/Qwen3.5-35B-A3B \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --guided-decoding-backend outlines \
    --quantization awq \
    --max-model-len 8192 \
    --port 8081
  ```
- **Alternative:** SGLang with native constrained decoding

**Migration from Claude API:**
- The `claude_client.py` Anthropic client is replaced with an OpenAI-compatible client
- Tool definitions (`TOOL_DEFINITIONS` in `claude_client.py`) translate directly — JSON schema format is universal
- Streaming response handling uses the same `async for` pattern
- Token budgeting tiers (256/1024/2048) map to `max_tokens` parameter
- Stop sequences transfer directly

### 2. Speech-to-Text Engine

**Model: Whisper large-v3-turbo (fine-tuned for aviation)**

| Property | Value |
|---|---|
| Base Model | whisper-large-v3-turbo (809M params) |
| Fine-tuning | LoRA on aviation audio corpus |
| Quantization | CTranslate2 INT8 |
| VRAM Required | ~2-3 GB |
| Latency | <500ms for 10s utterance on GPU |
| Server | fedirz/faster-whisper-server (existing) |

**Upgrade from current setup:**
- Current: `whisper-medium` (769M params, general purpose)
- New: `whisper-large-v3-turbo` (809M params, distilled from large-v3, aviation fine-tuned)
- large-v3-turbo is *faster* than medium (only 2 decoder layers vs 24) while being significantly more accurate
- Aviation fine-tuning improves recognition of: ATC phraseology, waypoint names, callsigns, ICAO codes, frequencies

**No architectural changes required** — same Docker container, same OpenAI-compatible API, same HTTP endpoint. Drop-in model swap.

### 3. Text-to-Speech Engine

**Primary: Kokoro v0.19+ (custom voice pack)**

| Property | Value |
|---|---|
| Parameters | ~82M |
| Architecture | StyleTTS2-based |
| VRAM Required | ~1-2 GB |
| Time-to-First-Audio | 50-150ms (GPU), 200-400ms (CPU) |
| Streaming | Yes, chunk-based |
| Voice Cloning | Via custom voice packs |
| License | Apache 2.0 |

**Voice Persona: F5-TTS zero-shot cloning**

For MERLIN's Navy test pilot voice, a two-stage approach:
1. **Kokoro** handles primary synthesis (fastest, highest quality for known voices)
2. **F5-TTS** provides zero-shot voice cloning from a 10-15 second reference clip of the target voice character (authoritative, clear, dry humor)
3. Once a satisfactory voice is identified, fine-tune Kokoro with 30-60 minutes of target voice audio for production quality

**TTS Serving:**
```
┌──────────────┐    HTTP POST     ┌──────────────┐
│ Orchestrator │ ───────────────► │  TTS Server  │
│  voice.py    │ text + voice_id  │  [:9091]     │
│              │ ◄─────────────── │  Kokoro      │
│              │  streaming audio │              │
└──────────────┘    (PCM/MP3)     └──────────────┘
```

**Migration from ElevenLabs:**
- Replace ElevenLabs REST/WebSocket calls with local HTTP endpoint
- Keep the existing TTS phrase cache (pre-generate common responses at startup)
- Keep the `tts_preprocessor.py` aviation text sanitizer (model-agnostic)
- Streaming architecture remains the same — Kokoro supports chunk-based streaming

### 4. Embedding Engine

**Model: bge-base-en-v1.5 (aviation fine-tuned)**

| Property | Value |
|---|---|
| Base Model | BAAI/bge-base-en-v1.5 (109M params) |
| Fine-tuning | TSDAE + contrastive on aviation corpus |
| Dimensions | 768 |
| VRAM Required | ~0.5 GB (or CPU) |
| Metric | Cosine similarity |

**Fine-tuning approach:**
1. **Stage 1 — TSDAE** (unsupervised): Feed all FAA handbooks, AIM, POH documents through denoising auto-encoder. Adapts embedding space to aviation terminology with zero labeled data.
2. **Stage 2 — Contrastive** (semi-supervised): 500-1,000 (query, positive passage, hard negative) triplets generated synthetically. Improves retrieval for aviation-specific queries like "engine failure after V1" matching documents about "rejected takeoff procedures."

**Integration:** ChromaDB already supports custom embedding functions. Swap the default model for the fine-tuned variant in `context_store.py`.

### 5. Local Airport Database

**Replace aviationapi.com with local SQLite**

The `lookup_airport` tool currently hits `https://api.aviationapi.com/v1/airports`. Replace with:
- Download FAA NASR 28-day subscription data (already partially done via `tools/download_faa_data.py`)
- Load into a local SQLite database
- Query directly from `tools.py` — eliminates network dependency and 10-second timeout

---

## Fine-Tuned Model Registry

All fine-tuned models are versioned and stored locally:

```
models/
├── llm/
│   ├── qwen3.5-35b-a3b-aviation-v1/     # Base + LoRA adapter
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── training_metadata.json
│   └── qwen3.5-35b-a3b-aviation-v1.gguf  # Merged + quantized for production
├── stt/
│   ├── whisper-large-v3-turbo-aviation/   # CTranslate2 format
│   └── training_metadata.json
├── tts/
│   ├── kokoro-merlin-voice-v1/            # Custom voice pack
│   └── f5-tts-merlin-reference.wav        # Reference audio for zero-shot
└── embeddings/
    ├── bge-base-aviation-v1/              # Fine-tuned sentence-transformer
    └── training_metadata.json
```

---

## Docker Compose (Local Inference)

```yaml
# docker-compose.local.yml
services:
  llm:
    image: vllm/vllm-openai:latest
    ports:
      - "8081:8000"
    volumes:
      - ./models/llm:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      --model /models/qwen3.5-35b-a3b-aviation-v1.gguf
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --guided-decoding-backend outlines
      --max-model-len 8192
      --gpu-memory-utilization 0.85

  whisper:
    image: fedirz/faster-whisper-server:latest-cuda
    ports:
      - "9090:8000"
    volumes:
      - ./models/stt:/models
    environment:
      - WHISPER__MODEL=/models/whisper-large-v3-turbo-aviation
      - WHISPER__INFERENCE_DEVICE=cuda

  tts:
    image: merlin-tts:latest  # Custom image with Kokoro
    ports:
      - "9091:9091"
    volumes:
      - ./models/tts:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data/chroma_db:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE

  orchestrator:
    build: ./orchestrator
    depends_on:
      - llm
      - whisper
      - tts
      - chromadb
    environment:
      - LLM_API_URL=http://llm:8000/v1
      - WHISPER_URL=http://whisper:8000
      - TTS_URL=http://tts:9091
      - CHROMA_URL=http://chromadb:8000
```

---

## VRAM Budget (Single RTX 5090, 32 GB)

| Component | VRAM | Notes |
|---|---|---|
| Qwen3.5-35B-A3B (Q4) | ~20 GB | Only 3B active params, but full weights in VRAM |
| Whisper large-v3-turbo (INT8) | ~2 GB | Loaded on-demand, can share GPU |
| Kokoro TTS | ~1 GB | Lightweight, always loaded |
| bge-base embeddings | ~0.5 GB | Can run on CPU instead |
| CUDA/framework overhead | ~2 GB | Drivers, context, buffers |
| **Total** | **~25.5 GB** | **Fits in 32 GB with 6.5 GB headroom** |

For a 24 GB card (RTX 4090), options:
- Run embeddings on CPU (saves 0.5 GB)
- Use Whisper medium instead of large-v3-turbo (saves ~0.5 GB)
- Use Q3_K_M quantization for LLM (saves ~3-4 GB, minor quality loss)
- All models still fit, but tighter

---

## API Compatibility Layer

To minimize code changes in the orchestrator, the local LLM exposes an **OpenAI-compatible API**. The migration in `claude_client.py` is:

```python
# BEFORE (Anthropic SDK)
self._client = anthropic.AsyncAnthropic(api_key=api_key)
async with self._client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=effective_max_tokens,
    system=system,
    messages=self._conversation,
    tools=TOOL_DEFINITIONS,
) as stream: ...

# AFTER (OpenAI SDK pointing at local vLLM)
self._client = openai.AsyncOpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed"
)
stream = await self._client.chat.completions.create(
    model="qwen3.5-35b-a3b-aviation",
    max_tokens=effective_max_tokens,
    messages=[{"role": "system", "content": system}] + self._conversation,
    tools=TOOL_DEFINITIONS,
    stream=True,
)
async for chunk in stream: ...
```

Tool definitions, tool result handling, and the agentic loop remain structurally identical.
