#!/bin/bash

# Start all services for local offline demo in separate Ptyxis terminal tabs.
# Waits for each service to be ready before starting the next.
# Usage: ./start.sh

# Shut down Steam if running (frees GPU VRAM for llama-server)
if pgrep -x steam > /dev/null 2>&1; then
    echo "Steam is running — shutting down to free GPU VRAM..."
    steam -shutdown
    while pgrep -x steam > /dev/null 2>&1; do
        sleep 1
    done
    echo "Steam shut down."
fi

wait_for_url() {
    local url=$1
    local name=$2
    local max_attempts=60
    local attempt=0
    echo "Waiting for $name at $url ..."
    while ! curl -sf "$url" > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "ERROR: $name did not start after ${max_attempts}s"
            exit 1
        fi
        sleep 1
    done
    echo "$name is ready."
}

# Tab 1: OKP (mimir) via podman
ptyxis --tab -T "OKP" -x "bash -c '
podman run --rm \
  --env \"ASK_RED_HAT_OFFLINE=true\" \
  --env \"INFERENCE_URL=http://localhost:8080/v1/\" \
  -p 8081:8080 -d --name okp mimir:latest \
  && podman logs -f okp;
exec bash
'"

wait_for_url "http://localhost:8081" "OKP"

# Tab 2: llama.cpp inference server
ptyxis --tab -T "llama-server" -x "bash -c '
cd ~/Downloads/llama.cpp
./build/bin/llama-server \
  -m ~/models/granite-4.1-8b/ibm-granite_granite-4.1-8b-Q4_K_M.gguf \
  -ngl 99 -fa on -np 1 \
  -c 16384 -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  --jinja --temp 0.3 --repeat-penalty 1.1 \
  --port 8082;
exec bash
'"

wait_for_url "http://localhost:8082/v1/models" "llama-server"

# Tab 3: lightspeed-stack
ptyxis --tab -T "lightspeed-stack" -x "bash -c '
cd ~/projects/lightspeed-stack
export EXTERNAL_PROVIDERS_DIR=../lightspeed-providers/resources/external_providers
uv run make run CONFIG=lightspeed-stack-local.yaml;
exec bash
'"

wait_for_url "http://localhost:8080/v1/models" "lightspeed-stack"

# Tab 4: verification
ptyxis --tab -T "verify" -x "bash -c '
echo \"=== llama-server models ===\"
curl -s http://localhost:8082/v1/models | python3 -m json.tool

echo \"\"
echo \"=== lightspeed-stack models ===\"
curl -s http://localhost:8080/v1/models | python3 -m json.tool

echo \"\"
echo \"=== test query ===\"
curl -sX POST http://localhost:8080/v1/query \
  -H \"Content-Type: application/json\" \
  -d \"{\\\"query\\\": \\\"configure remote desktop using gnome\\\"}\" | python3 -m json.tool 2>/dev/null || echo \"query failed\"

echo \"\"
echo \"All services running. This tab is free for testing.\"
exec bash
'"
