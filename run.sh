#!/usr/bin/env bash
# Start Whisper Studio using this project's virtual environment when available.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -x "faster-whisper-env/Scripts/python.exe" ]]; then
  PYTHON="faster-whisper-env/Scripts/python.exe"
elif [[ -x "faster-whisper-env/bin/python" ]]; then
  PYTHON="faster-whisper-env/bin/python"
else
  PYTHON="python"
fi

exec "$PYTHON" transcribe_web.py
