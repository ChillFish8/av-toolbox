#!/usr/bin/env bash
# Regenerates the lossless test media. Each file is 64 by 64, 8 frames, from ffmpeg's testsrc2 pattern.
set -euo pipefail
cd "$(dirname "$0")"

source_args=(-f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.32")

ffmpeg -y "${source_args[@]}" -pix_fmt yuv420p -c:v ffv1 yuv420p.mkv
ffmpeg -y "${source_args[@]}" -pix_fmt yuv420p10le -c:v ffv1 yuv420p10.mkv
ffmpeg -y "${source_args[@]}" -pix_fmt nv12 -c:v rawvideo nv12.nut
ffmpeg -y "${source_args[@]}" -pix_fmt rgb24 -c:v rawvideo rgb24.nut
ffmpeg -y "${source_args[@]}" -vf setfield=tff -pix_fmt yuv420p -c:v ffv1 -level 3 interlaced_tff.mkv
ffmpeg -y "${source_args[@]}" -vf setfield=bff -pix_fmt yuv420p -c:v ffv1 -level 3 interlaced_bff.mkv
