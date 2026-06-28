#!/bin/bash
# teslim/kod.zip'i repo kökünden tekrar üretir (yarışma dizin yapısı; D-2 §8).
# Kullanım: bash teslim/build_teslim.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)/teknofestv3"
mkdir -p "$STAGE"
for item in Dockerfile main.py requirements.txt README.md src roadguard config weights; do
  cp -R "$ROOT/$item" "$STAGE/"
done
find "$STAGE" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true
rm -f "$ROOT/teslim/kod.zip"
( cd "$(dirname "$STAGE")" && zip -qr "$ROOT/teslim/kod.zip" teknofestv3 -x '*.DS_Store' )
echo "kod.zip: $(du -h "$ROOT/teslim/kod.zip" | cut -f1)"
