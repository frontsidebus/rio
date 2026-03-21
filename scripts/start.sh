#!/usr/bin/env bash
# =============================================================================
# MERLIN Startup Script
# Starts all components: Docker services, SimConnect bridge, and web server.
# Run from WSL: ./scripts/start.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[MERLIN]${NC} $1"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $1"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $1"; }

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# --- 1. Docker services (Whisper + ChromaDB) --------------------------------
log "Starting Docker services..."
docker compose up -d whisper chromadb 2>/dev/null

# Wait for Whisper to be healthy
log "Waiting for Whisper to load model (this may take a minute on first run)..."
for i in $(seq 1 60); do
    status=$(docker inspect merlin-whisper --format '{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    if [ "$status" = "healthy" ]; then
        ok "Whisper STT ready"
        break
    fi
    if [ "$i" -eq 60 ]; then
        warn "Whisper still loading — continuing anyway (it will be ready soon)"
    fi
    sleep 5
done

# Check ChromaDB
if curl -sf http://localhost:8000/api/v2/heartbeat >/dev/null 2>&1; then
    ok "ChromaDB ready"
else
    warn "ChromaDB not responding yet — it should come up shortly"
fi

# --- 2. SimConnect Bridge (Windows .NET process) ----------------------------
log "Starting SimConnect bridge..."

# Kill any existing bridge instances
"/mnt/c/Windows/System32/taskkill.exe" /F /IM SimConnectBridge.exe >/dev/null 2>&1 || true
sleep 1

# Build and start in background
cd "$PROJECT_ROOT/simconnect-bridge"
"/mnt/c/Program Files/dotnet/dotnet.exe" build --verbosity quiet 2>/dev/null
"/mnt/c/Program Files/dotnet/dotnet.exe" run > "$PROJECT_ROOT/logs/bridge.log" 2>&1 &
BRIDGE_PID=$!
cd "$PROJECT_ROOT"

sleep 3
if kill -0 $BRIDGE_PID 2>/dev/null; then
    ok "SimConnect bridge started (PID: $BRIDGE_PID, log: logs/bridge.log)"
else
    warn "SimConnect bridge may have failed — check logs/bridge.log"
fi

# --- 3. Web server (FastAPI) ------------------------------------------------
log "Starting MERLIN web server..."

# Kill any existing web server
lsof -ti :3838 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1

# Activate venv and start
cd "$PROJECT_ROOT/web"
source "$PROJECT_ROOT/orchestrator/.venv/bin/activate"
python run.py > "$PROJECT_ROOT/logs/web.log" 2>&1 &
WEB_PID=$!
cd "$PROJECT_ROOT"

# Wait for web server to be ready
for i in $(seq 1 15); do
    if curl -sf http://localhost:3838/api/status >/dev/null 2>&1; then
        ok "Web server ready on http://localhost:3838"
        break
    fi
    if [ "$i" -eq 15 ]; then
        warn "Web server slow to start — check logs/web.log"
    fi
    sleep 2
done

# --- Summary ----------------------------------------------------------------
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  MERLIN AI Co-Pilot v1.1 — All Systems Go${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

# Check status
STATUS=$(curl -s http://localhost:3838/api/status 2>/dev/null)
if [ -n "$STATUS" ]; then
    SIM=$(echo "$STATUS" | python3 -c "import sys,json; print('CONNECTED' if json.load(sys.stdin).get('sim_connected') else 'WAITING')" 2>/dev/null || echo "?")
    WHISPER=$(echo "$STATUS" | python3 -c "import sys,json; print('OK' if json.load(sys.stdin).get('whisper_available') else 'DOWN')" 2>/dev/null || echo "?")
    CHROMA=$(echo "$STATUS" | python3 -c "import sys,json; print('OK' if json.load(sys.stdin).get('chromadb_available') else 'DOWN')" 2>/dev/null || echo "?")

    echo -e "  Cockpit UI:   ${GREEN}http://localhost:3838${NC}"
    echo -e "  SimConnect:   ${SIM}"
    echo -e "  Whisper STT:  ${WHISPER}"
    echo -e "  ChromaDB:     ${CHROMA}"
else
    echo -e "  Cockpit UI:   http://localhost:3838"
    echo -e "  (status check failed — server may still be starting)"
fi

echo ""
echo -e "  Logs:  tail -f logs/web.log"
echo -e "         tail -f logs/bridge.log"
echo ""
echo -e "  Stop:  ${CYAN}./scripts/stop.sh${NC}"
echo ""
