# MERLIN Local Inference — Data Flow

> **Status:** Proposal / RFC
> **Date:** 2026-03-22

## End-to-End Request Flow

This document traces every data path through the local inference stack, from pilot voice input to MERLIN audio response.

---

## 1. Voice Input Pipeline

```
┌───────────┐    Raw PCM     ┌──────────────┐   Cleaned WAV   ┌──────────────────┐
│ Microphone │──────────────►│  Audio Pre-   │───────────────►│  faster-whisper   │
│ (PTT/VAD)  │  16kHz mono   │  processor    │  high-pass,    │  large-v3-turbo   │
│            │               │              │  trimmed,       │  -aviation        │
└───────────┘               │  audio_      │  normalized     │                  │
                             │  processing  │                │  POST /v1/audio/ │
                             │  .py         │                │  transcriptions  │
                             └──────────────┘                └────────┬─────────┘
                                                                      │
                                                              JSON { "text": ... }
                                                                      │
                                                                      ▼
                                                             ┌────────────────┐
                                                             │  Orchestrator  │
                                                             │  (voice.py)    │
                                                             └────────────────┘
```

**Data transformations:**
1. **Capture:** Browser MediaRecorder → WebM opus (web) or PyAudio → PCM 16-bit (CLI)
2. **VAD:** Silero VAD detects speech boundaries (400ms silence timeout)
3. **Preprocessing:** High-pass filter (cockpit noise), silence trim (-40dB), peak normalize (95%)
4. **Transcription:** HTTP POST multipart/form-data to faster-whisper
   - Input: WAV audio bytes + `initial_prompt` (aviation vocabulary bias)
   - Output: `{"text": "check the ATIS at kilo juliet foxtrot kilo", "segments": [...]}`
5. **Confidence scoring:** `exp(avg_logprob)` from segments, warn below 0.4

---

## 2. LLM Reasoning Pipeline

```
                    ┌─────────────────────────────────────────────────────┐
                    │              LLM REQUEST CONSTRUCTION               │
                    │                                                     │
User text ────────► │  1. Classify query → select token budget           │
                    │     SHORT (256) / ROUTINE (1024) / BRIEFING (2048) │
                    │                                                     │
Sim telemetry ────► │  2. Build system prompt                            │
  (sim_client)      │     MERLIN persona + flight phase directive        │
                    │     + current sim state summary                     │
                    │                                                     │
Flight phase ─────► │  3. Assemble conversation history                  │
  (flight_phase)    │     Last 20 message pairs + new user message       │
                    │                                                     │
                    │  4. Attach tool definitions (5 tools)              │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                              OpenAI-compatible POST
                              /v1/chat/completions
                              (streaming, tools, max_tokens)
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │           vLLM / SGLang (GPU)                       │
                    │           Qwen3.5-35B-A3B-aviation                  │
                    │                                                     │
                    │  ┌─────────────┐    ┌──────────────────────┐       │
                    │  │ KV Cache    │    │ Outlines Constrained │       │
                    │  │ (PagedAttn) │    │ Decoding (tool JSON) │       │
                    │  └─────────────┘    └──────────────────────┘       │
                    │                                                     │
                    │  Output: streaming SSE chunks                       │
                    │    - text deltas (content)                          │
                    │    - tool_call deltas (function name + JSON args)   │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                   Stream chunks
                                           │
                                           ▼
```

---

## 3. Tool Execution Loop (Agentic)

```
                    ┌──────────────────────────────────────────────┐
                    │            AGENTIC TOOL LOOP                  │
                    │                                              │
                    │  while response contains tool_calls:         │
                    │                                              │
                    │    ┌──────────────────────────────────┐      │
                    │    │  Parse tool_call from stream     │      │
                    │    │  name: "get_sim_state"           │      │
                    │    │  args: {}                        │      │
                    │    └──────────┬───────────────────────┘      │
                    │               │                              │
                    │               ▼                              │
                    │    ┌──────────────────────────────────┐      │
                    │    │  Execute tool (tools.py)         │      │
                    │    │                                  │      │
                    │    │  get_sim_state ──► sim_client    │      │
                    │    │  lookup_airport ─► SQLite DB     │◄─── NEW: local DB
                    │    │  search_manual ──► ChromaDB      │      │
                    │    │  get_checklist ──► YAML files    │      │
                    │    │  create_flight_plan ► internal   │      │
                    │    └──────────┬───────────────────────┘      │
                    │               │                              │
                    │          Tool result JSON                    │
                    │               │                              │
                    │               ▼                              │
                    │    ┌──────────────────────────────────┐      │
                    │    │  Append to conversation:         │      │
                    │    │  {role: "tool", content: result} │      │
                    │    │                                  │      │
                    │    │  POST /v1/chat/completions again │      │
                    │    │  (with tool results in context)  │      │
                    │    └──────────────────────────────────┘      │
                    │                                              │
                    │  end loop → final text response              │
                    └──────────────────┬───────────────────────────┘
                                       │
                               Text response stream
                                       │
                                       ▼
```

---

## 4. TTS Output Pipeline

```
           Text response        ┌──────────────┐   Sanitized text  ┌──────────────┐
  ─────────────────────────────►│  TTS Pre-    │──────────────────►│  Kokoro TTS  │
    (streaming from LLM)        │  processor   │                   │  Server      │
                                │              │   "flight level   │  [:9091]     │
                                │  tts_pre-    │    three five     │              │
                                │  processor   │    zero"          │  POST /tts   │
                                │  .py         │                   │  streaming   │
                                └──────────────┘                   └──────┬───────┘
                                                                          │
  Transformations applied by TTS preprocessor:                    Audio chunks
  • FL350 → "flight level three five zero"                     (PCM/MP3 stream)
  • 118.30 → "one one eight point three zero"                         │
  • RWY 27L → "runway two seven left"                                │
  • SQ 7700 → "squawk seven seven zero zero"                         ▼
  • Strip markdown: **bold**, `code`, [links]               ┌──────────────┐
  • Remove special chars that garble speech                  │  Audio Out   │
                                                             │  (speaker/   │
                                                             │   WebSocket) │
                                                             └──────────────┘

  Latency budget:
  ┌─────────┬────────────┬──────────────┬────────────┬──────────────┐
  │  STT    │  LLM TTFT  │  TTS TTFA    │  Network   │  TOTAL       │
  │ ~300ms  │  ~200ms    │  ~100ms      │  ~0ms      │  ~600ms      │
  │         │            │  (Kokoro)    │  (local)   │  end-to-end  │
  └─────────┴────────────┴──────────────┴────────────┴──────────────┘
  vs. cloud: ~300ms STT + ~800ms LLM + ~400ms TTS + ~200ms net = ~1700ms
```

---

## 5. RAG Query Flow

```
  User query                 ┌──────────────────────────────────────────┐
  "What are the minimums     │          CONTEXT STORE                    │
   for the ILS 22L?"         │          (context_store.py)               │
        │                    │                                          │
        ▼                    │  1. Check query cache (TTL 60s)          │
  ┌──────────────┐           │     cache_key = SHA256(query + n + phase)│
  │ search_manual│           │                                          │
  │ tool called  │──────────►│  2. Flight-phase topic filter            │
  │ by LLM      │           │     APPROACH → ["approach", "ILS",       │
  └──────────────┘           │     "minimums", "go-around"]             │
                             │                                          │
                             │  3. Embed query                          │
                             │     bge-base-aviation-v1                 │
                             │     768-dim vector                       │
                             │                                          │
                             │  4. ChromaDB cosine search               │
                             │     n_results=5, where=phase_filter      │
                             │                                          │
                             │  5. Return ranked passages               │
                             └──────────────────────┬───────────────────┘
                                                    │
                                              Passage text
                                              + metadata
                                                    │
                                                    ▼
                                            Injected as tool
                                            result into LLM
                                            conversation
```

---

## 6. Telemetry & Flight Phase Flow

```
  ┌───────────────┐   SimConnect    ┌────────────────┐    WS JSON     ┌──────────────┐
  │  MSFS 2024    │───────────────►│  SimConnect     │──────────────►│  sim_client   │
  │  (Simulator)  │  event-driven   │  Bridge.exe     │  telemetry     │  .py          │
  │               │  message pump   │  (C# .NET 8)    │  updates       │               │
  └───────────────┘                │  [:8080]         │  (delta only)  │  Pydantic     │
                                    └────────────────┘                │  SimState     │
                                                                      └───────┬──────┘
                                                                              │
                                                              ┌───────────────┤
                                                              │               │
                                                              ▼               ▼
                                                    ┌──────────────┐  ┌──────────────┐
                                                    │ FlightPhase  │  │  Web UI      │
                                                    │ Detector     │  │  broadcast   │
                                                    │              │  │  to browsers │
                                                    │ State machine│  └──────────────┘
                                                    │ + hysteresis │
                                                    │ (3 consec.)  │
                                                    └──────┬───────┘
                                                           │
                                                    Phase change
                                                           │
                                              ┌────────────┼────────────┐
                                              ▼            ▼            ▼
                                        System prompt  RAG filter   Response
                                        style update   topic shift  style shift
```

---

## 7. Barge-In / Interruption Flow

```
  User speaks          ┌──────────────────────────────────────────┐
  while MERLIN         │         INTERRUPT HANDLER                 │
  is responding        │                                          │
       │               │  1. Set asyncio.Event (interrupt flag)   │
       ▼               │                                          │
  ┌──────────┐         │  2. Cancel active LLM stream             │
  │ New audio│────────►│     → vLLM: close SSE connection         │
  │ detected │         │     → Aborts generation server-side      │
  └──────────┘         │                                          │
                       │  3. Cancel active TTS stream             │
                       │     → Close Kokoro HTTP connection       │
                       │     → sounddevice.stop() / WS close      │
                       │                                          │
                       │  4. Process new input                    │
                       │     → STT on new audio                   │
                       │     → New LLM request                    │
                       └──────────────────────────────────────────┘

  Barge-in latency (local): ~50ms (event propagation only, no network round-trip)
  Barge-in latency (cloud): ~200-500ms (cancel in-flight API calls)
```

---

## 8. Complete Request Lifecycle (Happy Path)

```
  Time ──────────────────────────────────────────────────────────────────►

  t=0ms     t=300ms      t=500ms         t=600ms      t=700ms
  │         │            │               │            │
  │ ┌─────┐ │ ┌────────┐ │ ┌───────────┐ │ ┌────────┐ │ ┌──────────┐
  │ │ VAD │ │ │  STT   │ │ │ LLM gen   │ │ │TTS gen │ │ │ Audio    │
  │ │     │ │ │whisper │ │ │ + tool    │ │ │Kokoro  │ │ │ playback │
  │ │speak│ │ │large-  │ │ │ calls    │ │ │stream  │ │ │ begins   │
  │ │ end │ │ │v3-turbo│ │ │ (stream) │ │ │        │ │ │          │
  │ └─────┘ │ └────────┘ │ └──────┬──┘ │ └────────┘ │ └──────────┘
  │         │            │        │     │            │
  │         │            │  ┌─────▼───┐ │            │
  │         │            │  │get_sim_ │ │            │
  │         │            │  │state    │ │            │
  │         │            │  │(~5ms)   │ │            │
  │         │            │  └─────────┘ │            │
  │         │            │               │            │
  ▼         ▼            ▼               ▼            ▼

  Total end-to-end: ~700ms from speech end to audio playback start
  (vs. ~1700ms+ with cloud APIs)
```
