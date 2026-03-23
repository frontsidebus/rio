# MERLIN Local Inference — Hardware & Cost Estimate

> **Status:** Proposal / RFC
> **Date:** 2026-03-22

## Current Cloud Costs (Baseline)

Estimated monthly costs for the current cloud-dependent architecture:

| Service | Unit Cost | Estimated Usage | Monthly Cost |
|---|---|---|---|
| Anthropic Claude API (Sonnet) | $3/$15 per MTok in/out | ~2M in + 500K out tokens/mo | ~$13.50 |
| ElevenLabs TTS (Scale plan) | $99/mo | ~500K chars/mo | $99.00 |
| Docker hosting (Whisper + ChromaDB) | Self-hosted | Local compute | $0 |
| **Total recurring** | | | **~$112.50/mo** |

*Note: Usage estimates assume ~2 hours of active flight sim use per day, ~20 days/month.*

For heavier usage (8 hrs/day, every day):
- Claude API: ~$40-60/mo
- ElevenLabs: $99/mo (plan-based, not usage-based at Scale tier)
- **Heavy usage total: ~$140-160/mo**

---

## Hardware Recommendations

### Tier 1: Prosumer — Single GPU ($3,000 - $4,000)

**Best for:** 8B-class models, fast inference, single-user.

| Component | Selection | Est. Price |
|---|---|---|
| GPU | NVIDIA RTX 5090 32 GB GDDR7 | $2,000 |
| CPU | AMD Ryzen 7 9700X (8C/16T) | $320 |
| RAM | 64 GB DDR5-5600 (2×32 GB) | $180 |
| Storage | 2 TB NVMe Gen4 (WD SN850X) | $130 |
| Motherboard | MSI MAG B850 Tomahawk (AM5) | $200 |
| PSU | Corsair RM1000x (1000W, 80+ Gold) | $170 |
| Case | Fractal Design Meshify 2 | $130 |
| CPU Cooler | Noctua NH-D15 | $100 |
| **Total** | | **~$3,230** |

**What fits in 32 GB VRAM:**

| Model | VRAM | tok/s (est.) |
|---|---|---|
| Qwen3.5-35B-A3B Q4 (3B active) | ~20 GB | 60-100+ |
| + Whisper large-v3-turbo INT8 | +2 GB | real-time |
| + Kokoro TTS | +1 GB | <150ms TTFA |
| + bge-base embeddings | +0.5 GB | <50ms |
| + overhead | +2 GB | — |
| **Total** | **~25.5 GB** | **All concurrent** |

**Performance profile:**
- LLM: 60-100+ tok/s (MoE, only 3B active per token)
- STT: <500ms for 10s utterance
- TTS: 50-150ms time-to-first-audio
- End-to-end voice latency: ~600-800ms

**Cloud cost breakeven:** ~$112/mo ÷ 1 = **29 months** at current usage. With heavier usage (~$150/mo), breakeven at **22 months**.

**Limitation:** Running a dense 70B model is not possible. The MoE Qwen3.5-35B-A3B offers 35B-class quality with 3B inference cost, which is the sweet spot for this tier.

---

### Tier 2: Workstation — Dual GPU ($6,500 - $8,500)

**Best for:** 70B-class dense models, maximum quality, headroom for larger models.

| Component | Selection | Est. Price |
|---|---|---|
| GPU 1 | NVIDIA RTX 5090 32 GB GDDR7 | $2,000 |
| GPU 2 | NVIDIA RTX 5090 32 GB GDDR7 | $2,000 |
| CPU | AMD Threadripper 7960X (24C/48T) | $1,300 |
| RAM | 128 GB DDR5-5600 ECC (4×32 GB) | $400 |
| Storage | 2 TB NVMe Gen5 + 2 TB Gen4 | $280 |
| Motherboard | ASUS Pro WS TRX50-SAGE (sTR5) | $700 |
| PSU | Corsair HX1500i (1500W, 80+ Plat) | $350 |
| Case | be quiet! Dark Base Pro 901 | $260 |
| CPU Cooler | Noctua NH-U14S TR5-SP6 | $120 |
| **Total** | | **~$7,410** |

**What fits in 64 GB total VRAM:**

| Model | VRAM | tok/s (est.) |
|---|---|---|
| Llama 3.1 70B Q4_K_M (tensor parallel) | ~42 GB | 20-30 |
| OR Qwen3.5-35B-A3B Q8 (higher quality) | ~25 GB | 80+ |
| + Whisper large-v3-turbo | +2 GB | real-time |
| + Kokoro TTS | +1 GB | <150ms TTFA |
| + bge-base embeddings | +0.5 GB | <50ms |

**Why Threadripper:** 64 PCIe 5.0 lanes. Both GPUs run at full x16 bandwidth. Consumer AM5 would choke the second GPU to x8.

**Performance profile (70B Q4 tensor parallel):**
- LLM: 20-30 tok/s (may fall short of 30 tok/s target)
- All other services: identical to Tier 1

**Performance profile (Qwen3.5-35B-A3B Q8, recommended):**
- LLM: 80-120 tok/s (higher quality quantization + MoE efficiency)
- Dedicate GPU 2 entirely to STT/TTS with massive headroom
- This is the **recommended configuration** — Q8 MoE gives better quality than Q4 dense 70B

**Cloud cost breakeven:** ~28-35 months at $150-200/mo estimated cloud costs (higher API usage expected at this tier).

---

### Tier 3: Enterprise — Single H100 ($28,000 - $35,000)

**Best for:** Maximum performance, 70B+ models at full speed, future-proof.

| Component | Selection | Est. Price |
|---|---|---|
| GPU | NVIDIA H100 PCIe 80 GB HBM3 | $22,000 |
| CPU | AMD EPYC 9354 (32C/64T) | $2,200 |
| RAM | 256 GB DDR5-4800 ECC (8×32 GB) | $900 |
| Storage | 4 TB NVMe Gen5 RAID | $500 |
| Platform | Supermicro GPU workstation chassis | $3,500 |
| Networking | 10GbE (if serving multiple clients) | $200 |
| **Total** | | **~$29,300** |

**What fits in 80 GB HBM3:**

| Model | VRAM | tok/s (est.) |
|---|---|---|
| Llama 3.1 70B Q4_K_M | ~42 GB | 40-60+ |
| + Whisper large-v3-turbo | +2 GB | real-time |
| + Kokoro TTS | +1 GB | <100ms TTFA |
| + bge-base embeddings | +0.5 GB | <50ms |
| + overhead + headroom | ~34.5 GB free | — |
| **Total** | **~46 GB used** | **Everything concurrent** |

**Why H100:** 3,350 GB/s memory bandwidth (vs 1,792 GB/s on RTX 5090). LLM inference is memory-bandwidth-bound — H100 delivers 2x the tokens/second of consumer GPUs for the same model.

**Performance profile:**
- LLM (70B Q4): 40-60+ tok/s — comfortably exceeds 30 tok/s target
- LLM (Qwen3.5-35B-A3B Q8): 100-150+ tok/s
- All services fit on single GPU with 34 GB headroom
- Future-proof: room for 100B+ models or higher-quality quantization

**Alternative:** Lambda Hyperplane workstation pre-built with H100 (~$35,000-40,000, includes support).

---

## Cloud GPU Alternative (OpEx Model)

For teams that prefer operational expenditure over capital:

| Provider | GPU | $/hr | $/mo (8hr/day) | $/mo (24/7) |
|---|---|---|---|---|
| RunPod | RTX 4090 24 GB | $0.40 | $96 | $288 |
| RunPod | L40S 48 GB | $0.74 | $178 | $533 |
| RunPod | H100 SXM 80 GB | $2.49 | $598 | $1,793 |
| Vast.ai | RTX 4090 | $0.25 | $60 | $180 |
| Lambda | A100 80 GB | $1.29 | $310 | $929 |

**Recommendation:** RunPod RTX 4090 at $0.40/hr is excellent for development and testing. For production, owning hardware breaks even within 2 years.

---

## Cost Comparison Summary

| | Current (Cloud) | Tier 1 (Prosumer) | Tier 2 (Workstation) | Tier 3 (Enterprise) |
|---|---|---|---|---|
| **Upfront** | $0 | $3,230 | $7,410 | $29,300 |
| **Monthly (API/cloud)** | $112-160 | $0 | $0 | $0 |
| **Monthly (power)** | — | ~$15-25 | ~$25-40 | ~$40-60 |
| **Monthly (total)** | $112-160 | $15-25 | $25-40 | $40-60 |
| **Breakeven** | — | 22-29 months | 28-35 months | 24-30 months* |
| **LLM Quality** | Claude Sonnet (best) | MoE 35B (very good) | MoE 35B Q8 or 70B | 70B Q4+ (excellent) |
| **LLM Speed** | ~30-50 tok/s | 60-100+ tok/s | 80-120+ tok/s | 100-150+ tok/s |
| **TTS Quality** | ElevenLabs (best) | Kokoro (good) | Kokoro (good) | Kokoro (good) |
| **Privacy** | Data leaves network | Fully local | Fully local | Fully local |
| **Offline capable** | No | Yes | Yes | Yes |

*Enterprise breakeven assumes replacing higher cloud costs (~$500-1000/mo) for multi-user or heavy usage scenarios.*

---

## Recommended Path

### For MERLIN (single-user flight sim copilot):

**Tier 1 (RTX 5090, $3,230)** is the right choice.

Rationale:
1. Qwen3.5-35B-A3B MoE gives 35B reasoning quality at 3B inference cost — this is the breakthrough that makes single-GPU viable for high-quality aviation AI
2. 60-100+ tok/s exceeds the 30 tok/s target by 2-3x
3. All four models (LLM + STT + TTS + embeddings) fit concurrently in 32 GB
4. End-to-end voice latency drops from ~1700ms (cloud) to ~600ms (local)
5. Fully offline capable — works without internet
6. Breaks even with cloud costs in ~2 years
7. Fine-tuning on the same GPU (QLoRA fits in remaining VRAM)

**Upgrade path:** If the 35B MoE model proves insufficient for complex aviation reasoning, drop in a second RTX 5090 (Tier 2) and run a 70B dense model. The architecture supports this without code changes.

---

## Fine-Tuning Compute Costs

| Task | Hardware | Time (est.) | Cloud Cost (if rented) |
|---|---|---|---|
| LLM QLoRA (35B MoE, 2K examples) | 1× RTX 4090/5090 | 4-8 hours | ~$3-4 (RunPod) |
| Whisper LoRA (10 hrs audio) | 1× RTX 4090/5090 | 2-4 hours | ~$1-2 |
| Embedding fine-tuning (1K triplets) | 1× RTX 4090/5090 | 1-2 hours | ~$0.50-1 |
| TTS voice training (30 min audio) | 1× RTX 4090/5090 | 2-4 hours | ~$1-2 |
| **Total fine-tuning** | | **~10-18 hours** | **~$6-9** |

Fine-tuning is a one-time cost. On owned hardware, the only cost is electricity (~$2-3 for the full suite).

---

## Power & Thermal Considerations

| Tier | Peak GPU TDP | System Draw (est.) | Monthly Power (8hr/day) | Annual Power |
|---|---|---|---|---|
| Tier 1 (1× RTX 5090) | 575W | ~700W | ~$15-25 | ~$180-300 |
| Tier 2 (2× RTX 5090) | 1,150W | ~1,350W | ~$25-40 | ~$300-480 |
| Tier 3 (1× H100 PCIe) | 350W | ~600W | ~$12-20 | ~$144-240 |

*Based on $0.12-0.15/kWh US average residential rate.*

**Thermal notes:**
- RTX 5090: 3-slot cooler, needs good case airflow. Dual 5090s require a full tower with front-to-back airflow.
- H100 PCIe: Blower-style cooler, designed for enclosed chassis. Quieter than expected.
- All tiers: Air cooling is sufficient. Liquid cooling is optional and adds ~$200-400.
