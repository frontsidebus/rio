"""FastAPI backend for the MERLIN AI co-pilot web UI.

Bridges the browser frontend to the orchestrator components: SimConnect
telemetry streaming, Claude chat with the MERLIN persona, Whisper STT,
and ElevenLabs TTS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — make the orchestrator package importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from orchestrator.orchestrator.claude_client import ClaudeClient  # noqa: E402
from orchestrator.orchestrator.config import load_settings  # noqa: E402
from orchestrator.orchestrator.context_store import ContextStore  # noqa: E402
from orchestrator.orchestrator.flight_phase import FlightPhaseDetector  # noqa: E402
from orchestrator.orchestrator.sim_client import SimConnectClient, SimState  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("merlin.web")

# ---------------------------------------------------------------------------
# Shared application state (initialised in lifespan)
# ---------------------------------------------------------------------------
settings = load_settings()
logging.getLogger().setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

sim_client: SimConnectClient | None = None
claude_client: ClaudeClient | None = None
context_store: ContextStore | None = None
phase_detector: FlightPhaseDetector | None = None

# Track whether we have a live connection to the SimConnect bridge
_sim_connected: bool = False


# ---------------------------------------------------------------------------
# Lifespan — start / stop background services
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_client, claude_client, context_store, phase_detector, _sim_connected

    logger.info("Starting MERLIN web server")

    # Context store (ChromaDB) — degrades gracefully if unavailable
    context_store = ContextStore(chromadb_url=settings.chromadb_url)

    # SimConnect client
    sim_client = SimConnectClient(url=settings.simconnect_bridge_url)
    try:
        await sim_client.connect()
        _sim_connected = True
        logger.info("SimConnect bridge connected at %s", settings.simconnect_bridge_url)
    except Exception as exc:
        _sim_connected = False
        logger.warning(
            "SimConnect bridge unavailable at %s (%s); telemetry will be offline",
            settings.simconnect_bridge_url,
            exc,
        )

    # Flight phase detector
    phase_detector = FlightPhaseDetector()

    # Register the phase detector as a subscriber when connected
    if _sim_connected and sim_client is not None:
        async def _on_state(state: SimState) -> None:
            assert phase_detector is not None
            detected = phase_detector.update(state)
            state.flight_phase = detected

        sim_client.subscribe(_on_state)

    # Claude client
    claude_client = ClaudeClient(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
        sim_client=sim_client,
        context_store=context_store,
    )

    logger.info("MERLIN web server ready on port 3000")
    yield

    # Shutdown
    logger.info("Shutting down MERLIN web server")
    if _sim_connected and sim_client is not None:
        await sim_client.disconnect()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MERLIN AI Co-Pilot",
    description="Web backend for the MERLIN flight simulator co-pilot",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the frontend."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "MERLIN web UI — place index.html in web/static/"}


@app.get("/api/status")
async def get_status():
    """Return health status of all subsystems."""
    whisper_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.whisper_url}/")
            whisper_ok = resp.status_code < 500
    except Exception:
        pass

    chromadb_ok = context_store.available if context_store else False

    return {
        "sim_connected": _sim_connected,
        "chromadb_available": chromadb_ok,
        "chromadb_documents": context_store.document_count if context_store else 0,
        "whisper_available": whisper_ok,
        "elevenlabs_configured": bool(settings.elevenlabs_api_key and settings.voice_id),
        "claude_model": settings.claude_model,
        "simconnect_bridge_url": settings.simconnect_bridge_url,
    }


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile):
    """Transcribe uploaded audio via the Whisper Docker service.

    Accepts webm or wav from the browser MediaRecorder. Converts webm to wav
    via ffmpeg before forwarding to Whisper.
    """
    audio_bytes = await file.read()
    content_type = file.content_type or ""
    filename = file.filename or "audio.webm"

    # If the upload is webm, convert to wav with ffmpeg
    if "webm" in content_type or filename.endswith(".webm"):
        audio_bytes = await _convert_webm_to_wav(audio_bytes)
        filename = "audio.wav"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.whisper_url}/asr",
                files={"audio_file": (filename, audio_bytes, "audio/wav")},
                params={
                    "encode": "true",
                    "task": "transcribe",
                    "language": "en",
                    "output": "json",
                },
            )
            resp.raise_for_status()
            result = resp.json()
            text = result.get("text", "").strip()
            return {"text": text}
    except httpx.HTTPError as exc:
        logger.error("Whisper transcription failed: %s", exc)
        return {"text": "", "error": f"Transcription failed: {exc}"}


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech via ElevenLabs and return MP3 audio."""
    if not settings.elevenlabs_api_key or not settings.voice_id:
        return Response(
            content=json.dumps({"error": "ElevenLabs not configured"}),
            status_code=503,
            media_type="application/json",
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{settings.voice_id}",
                headers={
                    "xi-api-key": settings.elevenlabs_api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": request.text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
            )
            resp.raise_for_status()
            return Response(content=resp.content, media_type="audio/mpeg")
    except httpx.HTTPError as exc:
        logger.error("ElevenLabs TTS failed: %s", exc)
        return Response(
            content=json.dumps({"error": f"TTS failed: {exc}"}),
            status_code=502,
            media_type="application/json",
        )


# ---------------------------------------------------------------------------
# WebSocket: /ws/telemetry
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    """Stream simulator telemetry to the browser.

    Connects (or reconnects) to the SimConnect bridge on demand and
    proxies telemetry as JSON. Falls back to polling if the bridge
    subscriber model isn't active.
    """
    await ws.accept()
    logger.info("Telemetry WebSocket client connected")

    import websockets as ws_lib

    bridge_url = settings.simconnect_bridge_url

    try:
        while True:
            # Try to connect directly to the SimConnect bridge WebSocket
            try:
                async with ws_lib.connect(bridge_url) as bridge_ws:
                    logger.info("Telemetry proxy connected to bridge at %s", bridge_url)
                    await ws.send_json({"type": "telemetry", "connected": True, "data": None})

                    async for raw_msg in bridge_ws:
                        try:
                            data = json.loads(raw_msg)
                            # Detect flight phase
                            if phase_detector and "position" in data:
                                try:
                                    state = SimState.model_validate(data)
                                    fp = phase_detector.update(state)
                                    data["flight_phase"] = fp.value
                                except Exception:
                                    pass
                            await ws.send_json({
                                "type": "telemetry",
                                "connected": True,
                                "data": data,
                            })
                        except json.JSONDecodeError:
                            pass

            except (ConnectionRefusedError, OSError, Exception) as exc:
                logger.debug("Bridge not available (%s), retrying in 3s", exc)
                await ws.send_json({
                    "type": "telemetry",
                    "connected": False,
                    "flight_phase": "PREFLIGHT",
                    "data": None,
                })
                await asyncio.sleep(3.0)

    except WebSocketDisconnect:
        logger.info("Telemetry WebSocket client disconnected")
    except Exception as exc:
        logger.warning("Telemetry WebSocket error: %s", exc)


# ---------------------------------------------------------------------------
# WebSocket: /ws/chat
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """Chat with MERLIN.

    Receives JSON: {"text": "...", "audio_base64": "..."}
    If audio_base64 is present, transcribes it first via Whisper.
    Streams response as:
      {"type": "text", "content": "..."}   — streamed chunks
      {"type": "audio_url", "url": "..."}  — TTS URL for the full reply
      {"type": "done"}                     — signals end of response
    """
    await ws.accept()
    logger.info("Chat WebSocket client connected")

    pending_audio_mime: str | None = None

    try:
        while True:
            message = await ws.receive()

            # Handle binary audio data from the browser's MediaRecorder
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                logger.info("Received %d bytes of audio (mime: %s)", len(audio_bytes), pending_audio_mime)
                user_text = await _transcribe_audio_bytes(audio_bytes, pending_audio_mime or "audio/webm")
                pending_audio_mime = None
                if not user_text:
                    await ws.send_json({"type": "error", "content": "Could not transcribe audio"})
                    continue
                await ws.send_json({"type": "transcription", "text": user_text})
            elif "text" in message and message["text"]:
                raw = message["text"]
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "content": "Invalid JSON"})
                    continue

                # Handle audio_start marker (next message will be binary)
                if msg.get("type") == "audio_start":
                    pending_audio_mime = msg.get("mime", "audio/webm")
                    continue

                user_text = msg.get("text", "")
                if not user_text:
                    await ws.send_json({"type": "error", "content": "No text provided"})
                    continue
            else:
                continue

            # Stream Claude response with sentence-level TTS
            tts_enabled = bool(
                settings.elevenlabs_api_key and settings.voice_id
            )
            sentence_buffer = ""
            full_response = ""

            # TTS queue ensures audio chunks are sent in order
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def _tts_sender():
                """Sequentially synthesize and send TTS for queued sentences."""
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None:
                        break  # Poison pill — done
                    await _send_tts_chunk(ws, sentence)

            tts_task = asyncio.create_task(_tts_sender()) if tts_enabled else None

            try:
                assert claude_client is not None
                async for chunk in claude_client.chat(user_text):
                    full_response += chunk
                    await ws.send_json({"type": "text", "content": chunk})

                    if tts_enabled:
                        sentence_buffer += chunk
                        sent, remaining = _split_at_sentence(sentence_buffer)
                        if sent:
                            sentence_buffer = remaining
                            await tts_queue.put(sent)

                # Flush remaining text to TTS
                if tts_enabled and sentence_buffer.strip():
                    await tts_queue.put(sentence_buffer.strip())

            except Exception as exc:
                logger.exception("Claude chat error")
                await ws.send_json({"type": "error", "content": f"Chat error: {exc}"})
                if tts_task:
                    await tts_queue.put(None)
                    await tts_task
                continue

            await ws.send_json({"type": "done"})

            # Signal TTS sender to finish and wait for it
            if tts_task:
                await tts_queue.put(None)
                await tts_task

    except WebSocketDisconnect:
        logger.info("Chat WebSocket client disconnected")
    except Exception as exc:
        logger.warning("Chat WebSocket error: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_at_sentence(text: str) -> tuple[str, str]:
    """Split text at the last sentence boundary. Returns (complete, remaining)."""
    # Look for sentence endings followed by a space (to avoid splitting mid-abbreviation)
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?\n" and (i + 1 >= len(text) or text[i + 1] == " " or text[i + 1] == "\n"):
            return text[: i + 1].strip(), text[i + 1 :].lstrip()
    return "", text  # No sentence boundary found yet


# Shared httpx client for TTS (avoid creating per-request)
_tts_client: httpx.AsyncClient | None = None


async def _get_tts_client() -> httpx.AsyncClient:
    global _tts_client
    if _tts_client is None or _tts_client.is_closed:
        _tts_client = httpx.AsyncClient(timeout=30.0)
    return _tts_client


async def _send_tts_chunk(ws: WebSocket, text: str) -> None:
    """Synthesize a sentence and send the audio back over WebSocket as a binary message."""
    try:
        client = await _get_tts_client()
        resp = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{settings.voice_id}/stream",
            headers={
                "xi-api-key": settings.elevenlabs_api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.3,
                },
            },
        )
        resp.raise_for_status()
        # Send audio as binary WebSocket frame — browser will play it
        await ws.send_json({"type": "tts_audio", "size": len(resp.content)})
        await ws.send_bytes(resp.content)
    except Exception as exc:
        logger.warning("TTS chunk failed: %s", exc)


async def _convert_webm_to_wav(webm_bytes: bytes) -> bytes:
    """Convert webm audio to wav using ffmpeg in a subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src:
        src.write(webm_bytes)
        src_path = src.name

    dst_path = src_path.replace(".webm", ".wav")

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", dst_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("ffmpeg conversion failed: %s", stderr.decode(errors="replace"))
            return webm_bytes  # fall back to raw bytes

        return Path(dst_path).read_bytes()
    except FileNotFoundError:
        logger.warning("ffmpeg not found; sending raw webm to Whisper")
        return webm_bytes
    finally:
        # Clean up temp files
        for p in (src_path, dst_path):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


async def _transcribe_audio_bytes(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """Convert browser audio to wav and send to Whisper."""
    if "webm" in mime_type or "ogg" in mime_type:
        audio_bytes = await _convert_webm_to_wav(audio_bytes)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.whisper_url}/asr",
                files={"audio_file": ("audio.wav", audio_bytes, "audio/wav")},
                params={"encode": "true", "task": "transcribe", "language": "en", "output": "json"},
            )
            resp.raise_for_status()
            return resp.json().get("text", "").strip()
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        return ""


async def _transcribe_base64_audio(audio_b64: str) -> str:
    """Decode base64 audio and send to Whisper for transcription."""
    import base64

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        logger.warning("Failed to decode base64 audio")
        return ""

    # Assume webm from browser MediaRecorder; convert to wav
    audio_bytes = await _convert_webm_to_wav(audio_bytes)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.whisper_url}/asr",
                files={"audio_file": ("audio.wav", audio_bytes, "audio/wav")},
                params={
                    "encode": "true",
                    "task": "transcribe",
                    "language": "en",
                    "output": "json",
                },
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("text", "").strip()
    except Exception as exc:
        logger.error("Whisper transcription (base64) failed: %s", exc)
        return ""
