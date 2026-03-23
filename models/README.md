# Local Model Storage

This directory holds model weights for the local inference stack.
These files are git-ignored due to their size (multi-GB).

## Directory Structure

```
models/
├── llm/          # LLM model files (HuggingFace format)
│                 # Default: Qwen3.5-35B-A3B fine-tuned for aviation
├── stt/          # Speech-to-text models (CTranslate2 format)
│                 # Default: faster-whisper large-v3-turbo
├── tts/          # TTS voice packs and model weights
│                 # Default: Kokoro with MERLIN voice pack
└── embeddings/   # Sentence-transformer models for ChromaDB
                  # Default: all-MiniLM-L6-v2
```

## Setup

1. Download or fine-tune model weights into the appropriate subdirectory.
2. The `docker-compose.local.yml` mounts these directories into their containers.
3. Ensure file permissions allow the container user to read the model files.

## Storage Requirements

| Model                  | Approximate Size |
|------------------------|------------------|
| Qwen3.5-35B-A3B       | ~20 GB           |
| faster-whisper large-v3-turbo | ~3 GB    |
| Kokoro TTS             | ~1 GB           |
| Sentence-transformers  | ~100 MB         |
