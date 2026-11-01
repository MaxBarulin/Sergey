#!/usr/bin/env bash
# Разбивает исходник на кадры (JPEG, нумерация с 0) + трек красного маркера на карте.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/frames
ffmpeg -v error -y -i source/IMG_7435.MP4 -q:v 2 -start_number 0 work/frames/%05d.jpg
echo "frames: $(ls work/frames | wc -l)"
