# MERLIN Local Inference — Implementation Plan

> **Status:** Proposal / RFC
> **Date:** 2026-03-22

## Phased Implementation Strategy

The migration from cloud to local inference is broken into 4 phases. Each phase is independently deployable and adds value without requiring subsequent phases.

---

## Phase 1: Foundation (Weeks 1-3)

**Goal:** Local LLM inference with tool calling, replacing Claude API.

### 1.1 Abstract the LLM Client

Create an `LLMClient` protocol/interface that both the current Anthropic client and a new OpenAI-compatible client can implement.

```
orchestrator/
  orchestrator/
    llm/
      __init__.py
      base.py          # LLMClient protocol: stream(), tool handling
      anthropic.py     # Current claude_client.py, refactored
      openai_compat.py # New: OpenAI-compatible client for local models
```

**Key interface:**
- `async stream(messages, tools, max_tokens, system) -> AsyncIterator[StreamEvent]`
- `StreamEvent` unifies text deltas, tool calls, and stop events across both backends
- Config selects backend: `LLM_BACKEND=anthropic|local`

**Why abstract first:** This lets us A/B test cloud vs. local without code duplication. The orchestrator, tools, and voice pipeline don't change at all.

### 1.2 Deploy Local LLM Server

- Add vLLM service to `docker-compose.local.yml`
- Model: Start with **Qwen3.5-35B-A3B** (unmodified, pre-fine-tuning)
- Enable tool calling: `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
- Enable constrained decoding: `--guided-decoding-backend outlines`
- Validate all 5 tools work: `get_sim_state`, `lookup_airport`, `search_manual`, `get_checklist`, `create_flight_plan`

### 1.3 Validate Tool Calling Parity

Create a test harness that replays recorded conversations through both Claude and the local model, comparing:
- Tool selection accuracy (does it call the right tool?)
- Argument formatting (valid JSON, correct field names?)
- Response quality (aviation accuracy, persona consistency)
- Latency (time-to-first-token, total generation time)

**Exit criteria:** Local model correctly selects tools in >90% of test scenarios.

### 1.4 Localize Airport Lookup

- Extend `tools/download_faa_data.py` to build a SQLite database from OurAirports + FAA NASR data
- Replace HTTP call to `api.aviationapi.com` with SQLite query in `tools.py`
- Zero network dependency for airport lookups
- Include: airports, runways, frequencies, navaids

---

## Phase 2: Voice Pipeline (Weeks 3-5)

**Goal:** Replace ElevenLabs with local TTS, upgrade Whisper model.

### 2.1 Upgrade Whisper Model

Minimal-effort, high-impact change:
- Swap Docker config: `WHISPER__MODEL=large-v3-turbo` (currently `medium`)
- No code changes — same API, same Docker image
- Expected: better accuracy at equal or faster speed (2 decoder layers vs 24)
- Validate with aviation phrase test set

### 2.2 Deploy Local TTS (Kokoro)

- Build a Kokoro TTS Docker image with HTTP API
- API contract: `POST /tts` with `{text, voice_id}` → streaming audio chunks
- Evaluate built-in voice packs for MERLIN persona fit
- If no built-in voice works: use F5-TTS for zero-shot cloning from reference audio

### 2.3 Integrate TTS into Voice Pipeline

- Replace ElevenLabs REST calls in `voice.py` with local Kokoro HTTP calls
- Replace ElevenLabs WebSocket streaming in `server.py` with local streaming
- Keep the TTS phrase cache — regenerate cached phrases with new voice at startup
- Keep `tts_preprocessor.py` — it's model-agnostic

### 2.4 Build TTS Serving Container

```dockerfile
FROM python:3.11-slim
RUN pip install kokoro-tts torch
COPY models/tts /models
COPY tts_server.py /app/
CMD ["python", "/app/tts_server.py", "--port", "9091"]
```

Expose a simple FastAPI server with:
- `POST /tts` — single utterance, returns streaming audio
- `POST /tts/cache` — pre-generate and cache common phrases
- `GET /voices` — list available voice packs

**Exit criteria:** End-to-end voice interaction works entirely locally. Time-to-first-audio <500ms.

---

## Phase 3: Fine-Tuning (Weeks 5-10)

**Goal:** Domain-specialized models for aviation accuracy.

### 3.1 LLM Fine-Tuning Dataset

**Data generation pipeline:**

1. **Scenario templates** (200 scenarios):
   - 9 flight phases × ~22 scenarios each
   - Variables: aircraft type, airports, weather, pilot skill level, situation (normal/abnormal/emergency)

2. **Synthetic conversation generation:**
   - Use Claude Opus to generate multi-turn conversations from scenarios
   - Include tool calls with realistic tool responses
   - Inject MERLIN persona and flight-phase-appropriate style
   - Target: 2,000-3,000 conversations

3. **Human review:**
   - Aviation accuracy review by qualified pilot or instructor
   - Tool call correctness verification
   - Persona consistency check
   - Target: 100-200 reviews/day per reviewer

4. **Training data format:**
   ```json
   {
     "messages": [
       {"role": "system", "content": "<MERLIN persona + phase>"},
       {"role": "user", "content": "What's our fuel state?"},
       {"role": "assistant", "tool_calls": [
         {"name": "get_sim_state", "arguments": "{}"}
       ]},
       {"role": "tool", "name": "get_sim_state", "content": "{...}"},
       {"role": "assistant", "content": "Captain, showing 42 gallons..."}
     ]
   }
   ```

### 3.2 LLM Fine-Tuning Execution

- **Framework:** Unsloth (2x speed, 60% less VRAM)
- **Method:** QLoRA (4-bit base model + LoRA adapters)
- **LoRA config:** rank=16, alpha=32, target all attention + MLP projections
- **Hardware:** Single RTX 4090/5090 sufficient for QLoRA on 35B MoE model
- **Training:** ~1-3 epochs, 3e-4 learning rate, 8 effective batch size
- **Validation:** Hold out 10% of conversations for eval
- **Export:** Merge adapters → GGUF quantize → deploy to vLLM

### 3.3 Whisper Fine-Tuning

**Data collection (target: 10-20 hours):**

| Source | Hours | Notes |
|---|---|---|
| LiveATC recordings (transcribed) | 5-10 | ATC phraseology, callsigns |
| Self-recorded aviation phrases | 2-3 | Targeted vocab coverage |
| TTS-synthesized + noise augmented | 3-5 | Cockpit noise overlay |
| ATCOSIM corpus | ~10 | Free research dataset |

**Fine-tuning:**
- HuggingFace Transformers + PEFT (LoRA on encoder)
- Convert fine-tuned model to CTranslate2 format for faster-whisper
- Validate against aviation phrase test set (compare WER before/after)

### 3.4 TTS Voice Fine-Tuning

**Option A (quick): Zero-shot voice cloning**
- Record or source 15-30 seconds of target voice (Navy pilot character)
- Use F5-TTS for zero-shot cloning
- Evaluate output quality

**Option B (production): Kokoro fine-tuning**
- Script 500-1000 MERLIN-style sentences
- Record with voice actor (30-60 minutes, studio quality)
- Fine-tune Kokoro on recordings
- Export as custom voice pack

### 3.5 Embedding Fine-Tuning

**Stage 1 — TSDAE (unsupervised, ~2 hours):**
- Feed all ingested FAA documents through denoising auto-encoder
- Adapts embedding space to aviation terminology
- Zero labeled data required

**Stage 2 — Contrastive (semi-supervised, ~4 hours):**
- Generate 1,000 (query, positive passage, hard negative) triplets
- Use Claude to generate pilot questions from document passages
- Hard negative mining from Stage 1 model
- Train with triplet loss using `sentence-transformers`

**Exit criteria:** Measurable improvement on aviation retrieval benchmarks (precision@5 on held-out query set).

---

## Phase 4: Production Hardening (Weeks 10-14)

**Goal:** Reliability, monitoring, and operational maturity.

### 4.1 Health Monitoring

Extend existing health monitoring to cover all local inference services:

```
┌─────────────────────────────────────────────┐
│              HEALTH DASHBOARD                │
│                                             │
│  LLM Server    [████████████] UP  42 tok/s  │
│  Whisper STT   [████████████] UP  <300ms    │
│  Kokoro TTS    [████████████] UP  <150ms    │
│  ChromaDB      [████████████] UP  <50ms     │
│  SimConnect    [████████████] UP  20Hz      │
│  GPU VRAM      [████████░░░░] 25.5/32 GB    │
│  GPU Temp      [██████░░░░░░] 72°C          │
└─────────────────────────────────────────────┘
```

### 4.2 Graceful Degradation

Maintain current degradation patterns:
- LLM unavailable → text-only mode with cached responses
- STT unavailable → text input only
- TTS unavailable → text output only (no audio)
- ChromaDB unavailable → RAG returns empty (tools still work)
- **New:** GPU memory pressure → shed TTS to CPU, reduce LLM context length

### 4.3 Model Versioning & Rollback

```
models/
  llm/
    qwen3.5-35b-a3b-aviation-v1.gguf  # current
    qwen3.5-35b-a3b-aviation-v2.gguf  # candidate
    active -> v1                        # symlink
  stt/
    whisper-large-v3-turbo-aviation-v1/
    active -> v1
```

- Config points to `active` symlink
- Rollback = change symlink + restart service
- A/B testing via config flag

### 4.4 Benchmark Suite

Automated benchmarks run on every model update:

| Benchmark | Metric | Target |
|---|---|---|
| Tool selection accuracy | % correct tool chosen | >95% |
| Tool argument validity | % parseable JSON | 100% (constrained decoding) |
| Aviation knowledge (QA set) | % correct answers | >85% |
| Persona consistency | Human eval score 1-5 | >4.0 |
| LLM throughput | tokens/second | >30 |
| STT word error rate | WER on aviation test set | <10% |
| TTS naturalness | MOS score | >3.5 |
| End-to-end latency | speech-end to audio-start | <800ms |
| VRAM usage | peak GB | <30 GB (RTX 5090) |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Qwen3.5-35B-A3B tool calling quality below Claude | Medium | High | Fallback to Claude API via config flag; fine-tuning closes gap |
| VRAM exhaustion under concurrent load | Low | High | Monitor GPU memory; shed TTS to CPU as pressure valve |
| Fine-tuning produces hallucinations on aviation facts | Medium | High | Extensive eval suite; human review of training data; keep RAG as ground truth |
| Kokoro voice quality below ElevenLabs | Medium | Medium | Keep ElevenLabs as optional backend; invest in voice fine-tuning |
| Whisper fine-tuning degrades general recognition | Low | Medium | Evaluate on both aviation and general test sets; LoRA keeps base model intact |
| MoE model has inconsistent quality across experts | Low | Medium | Benchmark extensively; fall back to dense 9B if needed |

---

## Migration Checklist

### Phase 1 Deliverables
- [ ] `LLMClient` abstraction layer with Anthropic + OpenAI backends
- [ ] vLLM Docker service with Qwen3.5-35B-A3B
- [ ] All 5 tools validated with local model
- [ ] Local airport SQLite database
- [ ] Tool calling parity test harness
- [ ] Config flag: `LLM_BACKEND=anthropic|local`

### Phase 2 Deliverables
- [ ] Whisper large-v3-turbo deployed (drop-in upgrade)
- [ ] Kokoro TTS Docker service
- [ ] TTS integrated into voice pipeline and web server
- [ ] End-to-end voice loop working locally
- [ ] Config flag: `TTS_BACKEND=elevenlabs|local`

### Phase 3 Deliverables
- [ ] 2,000+ aviation conversation training dataset
- [ ] Fine-tuned LLM adapter (QLoRA)
- [ ] Fine-tuned Whisper model (aviation LoRA)
- [ ] Custom MERLIN TTS voice
- [ ] Fine-tuned embedding model
- [ ] Eval benchmarks for all fine-tuned models

### Phase 4 Deliverables
- [ ] Health monitoring for all local services
- [ ] GPU resource monitoring and alerting
- [ ] Model versioning and rollback system
- [ ] Automated benchmark suite
- [ ] Production deployment documentation
- [ ] `docker-compose.local.yml` fully operational
