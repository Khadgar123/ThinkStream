#!/usr/bin/env bash
set -euo pipefail

VIDEO_DIR="${VIDEO_DIR:-/home/tione/notebook/gaozhenkun/hzh/data/Video-MME/data}"
OUT_DIR="${OUT_DIR:-/home/tione/notebook/gaozhenkun/hzh/data/Video-MME/frames_2fps}"
JOBS="${JOBS:-16}"
FPS="${FPS:-2}"

mkdir -p "$OUT_DIR/_logs" "$OUT_DIR/_done"

extract_one() {
  local video="$1"
  local name
  name="$(basename "$video")"
  name="${name%.*}"

  local frame_dir="$OUT_DIR/$name"
  local done_file="$OUT_DIR/_done/$name.done"
  local log_file="$OUT_DIR/_logs/$name.log"

  if [[ -s "$done_file" ]]; then
    return 0
  fi

  mkdir -p "$frame_dir"
  ffmpeg -hide_banner -loglevel error -y -threads 1 \
    -i "$video" -vf "fps=$FPS" -q:v 2 "$frame_dir/%06d.jpg" \
    >"$log_file" 2>&1

  local count
  count="$(find "$frame_dir" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
  if [[ "$count" -gt 0 ]]; then
    printf '%s\t%s\n' "$video" "$count" >"$done_file"
  else
    printf 'no frames extracted from %s\n' "$video" >>"$log_file"
    return 1
  fi
}

export VIDEO_DIR OUT_DIR FPS
export -f extract_one

find "$VIDEO_DIR" -maxdepth 1 -type f -name '*.mp4' -print0 \
  | xargs -0 -n 1 -P "$JOBS" bash -c 'extract_one "$0"'
