#!/usr/bin/env bash
# Run the full benchmark matrix. Writes raw output to results/.
set -u
mkdir -p results
# Build first (expects build.cmd to be present)
cmd.exe //c ".\\build.cmd" >/dev/null 2>&1

# Quick smoke
echo "=== smoke 5K/12ts ==="
./wavefront_mc.exe net_small 2>&1 | tee results/run_5k_12ts.txt

# Full scaling runs — 24 timesteps
for size in 1k 10k 100k 309k; do
    echo "=== scaling $size @ 24ts ==="
    ./wavefront_mc.exe net_$size 2>&1 | tee results/run_${size}_24ts.txt
done

# Large runs — fewer timesteps so CPU doesn't dominate wall clock
for size in 500k_4 1m_4; do
    echo "=== scaling $size ==="
    ./wavefront_mc.exe net_$size 2>&1 | tee results/run_${size}.txt
done

echo "=== done ==="
