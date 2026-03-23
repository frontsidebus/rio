# RIO — Synthetic Voice Creation for MERLIN (No Voice Actors)

> **Status:** Research Complete / Ready for Implementation
> **Date:** 2026-03-22

## Problem Statement

MERLIN needs a distinctive Navy Test Pilot voice — authoritative, clear, calm under pressure, with dry humor. The constraint: **no voice actors**. The solution must be fully synthetic.

---

## Approaches Evaluated

### 1. Zero-Shot Cloning from NASA/Military Archives

Use 10-15 seconds of public domain NASA mission audio (Apollo/Shuttle CapCom) as a reference clip for F5-TTS or Fish Speech zero-shot cloning.

**Sources:** Apollo CapCom voices (Charlie Duke, Jack Lousma, Joe Kerwin) have the perfect cadence — calm, authoritative, clipped military comms. Shuttle-era audio has better fidelity.

**Challenge:** Apollo-era audio is heavily compressed (300-3000 Hz bandwidth, static). Zero-shot models clone the noise artifacts along with the voice. Mitigation: run through speech enhancement (Resemble Enhance, VoiceFixer) before cloning.

| Quality | Latency | Effort | Legal | Verdict |
|---|---|---|---|---|
| 2-3/5 | None (precomputed) | 1-2 days | Clean (US gov public domain) | Experiment / seed data |

### 2. Parler-TTS Text-Described Voice Design

Generate a voice from a natural language description: *"A middle-aged male with a deep, authoritative voice speaks clearly and calmly, with slight gravel and a confident cadence."*

**Strengths:** Fully local, zero reference audio needed, iterative prompt engineering to dial in timbre.
**Weaknesses:** Generic audiobook narrator quality, no military cadence, inconsistent across runs. Best as a **voice design workbench** to create clean reference clips for other models.

| Quality | Latency | Effort | Legal | Verdict |
|---|---|---|---|---|
| 2.5/5 (final) / 4/5 (as seed) | Too slow for prod (2-4x RT) | 2-4 hours | Clean (Apache 2.0) | Voice design tool, not production engine |

### 3. Cloud TTS Bootstrapping into Kokoro (RECOMMENDED)

Use Azure Neural TTS or Google Cloud TTS to generate 30-45 minutes of MERLIN dialogue, then fine-tune a Kokoro voice pack on that data.

**Why this wins:** Cloud TTS produces studio-quality training data. Kokoro fine-tuning creates a permanent local voice pack. Result runs at 10-50x realtime — fastest available. One-time cloud cost (~$15-20), then fully local forever.

**Legal:** Azure and Google explicitly license TTS output for any use. Clean.

| Quality | Latency | Effort | Legal | Verdict |
|---|---|---|---|---|
| 4/5 | Zero (Kokoro native speed) | 3-4 days | Clean (Azure/Google) | **Primary recommendation** |

### 4. Voice Embedding Interpolation

Blend Kokoro speaker embeddings arithmetically (e.g., 70% `am_adam` + 30% warmer voice) to create a custom blend.

**Best for:** Fine-tuning the voice *after* another approach establishes a baseline. Not a standalone solution.

| Quality | Latency | Effort | Legal | Verdict |
|---|---|---|---|---|
| 3/5 | Zero (precomputed) | 1-2 days | Clean | Refinement tool |

### 5. RVC Voice Conversion Pipeline

Generate neutral TTS then convert to MERLIN's voice in real-time via RVC.

**Rejected:** Adds 150-300ms latency per chunk, partially negating the streaming TTS architecture. Too much pipeline complexity for a cockpit-latency-sensitive application.

| Quality | Latency | Effort | Legal | Verdict |
|---|---|---|---|---|
| 4/5 | **+150-300ms (deal-breaker)** | 1-2 weeks | Clean | Not recommended for RIO |

### 6. Kokoro Stock Voices as Base

Evaluate built-in Kokoro voices. `am_adam` is the closest — deep, steady, clear. Lacks personality but is a strong fine-tuning base.

| Quality | Latency | Effort | Legal | Verdict |
|---|---|---|---|---|
| 2.5/5 (stock) / 4/5 (fine-tuned) | Best in class | 2 hours eval | Clean (Apache 2.0) | Use as inference engine + fine-tuning base |

---

## Recommended Implementation Plan

### Week 1: Fast Path (Parler-TTS + F5-TTS)

**Goal:** Working MERLIN voice in 2 days.

1. **Day 1:** Use Parler-TTS to design the voice. Iterate on text descriptions until the timbre matches. Select the best 15-second clip.
2. **Day 2:** Use that clip as F5-TTS zero-shot reference. Evaluate output quality. This becomes the interim MERLIN voice.

### Week 2: Production Path (Cloud Bootstrap + Kokoro)

**Goal:** Production-quality voice on the fastest engine.

1. **Day 3:** Script 200-400 MERLIN utterances covering the full register:
   - Terse callouts: "V1, rotate", "Gear up", "Roger, turning base"
   - Briefings: approach plates, weather, fuel state
   - Checklists: before takeoff, cruise, before landing
   - Casual: "Coffee's on me if we nail this crosswind", "Not my first rodeo, Captain"
2. **Day 4:** Generate all utterances via Azure Neural TTS with a selected authoritative male voice
3. **Day 5-6:** Fine-tune Kokoro voice pack on the 30-45 minutes of generated audio
4. **Day 7:** Integration test — swap into RIO TTS pipeline, validate latency and quality

### Optional: NASA Archive Experiment

In parallel, source Apollo CapCom clips and test Fish Speech zero-shot cloning with speech enhancement preprocessing. If the result has the right character, it could become the final voice identity — derived from actual test pilot audio.

---

## Cost Summary

| Item | Cost | Notes |
|---|---|---|
| Azure Neural TTS (data generation) | ~$15-20 | One-time, 30-45 min of audio |
| Parler-TTS inference | $0 | Local, Apache 2.0 |
| Kokoro fine-tuning compute | ~$2-3 electricity | 2-4 hours on consumer GPU |
| F5-TTS zero-shot | $0 | Local |
| **Total** | **~$17-23** | **One-time cost, fully local after** |

---

## Key Decision: Kokoro as Production TTS Engine

Regardless of voice creation approach, **Kokoro is the production TTS engine for RIO**:
- 10-50x realtime inference speed (50-150ms TTFA on GPU)
- ~82M parameters, ~1-2 GB VRAM
- Streaming chunk output
- Apache 2.0 license
- Fine-tunable voice packs

All voice creation approaches ultimately produce training data or reference audio that gets distilled into a Kokoro voice pack.
