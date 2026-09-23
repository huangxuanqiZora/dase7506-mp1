"""Measure the three evaluation limits for the frozen final predictor on CPU.

    CPU scoring time vs the classroom baseline (same machine, 4 threads, fp32)
    peak process RAM during scoring
    uncompressed inference asset size (checkpoint)

Usage:
    python measure_resources.py --checkpoint ../final/checkpoint.pt
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import psutil
import torch

from common import ROOT, load_data, make_model, setup
from evaluate import score


def timed_score(model, tokens, device, proc):
    peak = {'rss': 0}
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            peak['rss'] = max(peak['rss'], proc.memory_info().rss)
            time.sleep(0.05)

    thread = threading.Thread(target=monitor)
    thread.start()
    result = score(model, *tokens, device, 'fp32')
    stop.set()
    thread.join()
    return result, peak['rss']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()

    device, _ = setup('cpu', 'fp32', args.threads)
    data = load_data()
    validation = data['validation']
    proc = psutil.Process(os.getpid())

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    final_model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    final_model.load_state_dict(checkpoint['model'])
    baseline_cfg = json.loads((ROOT / 'configs/baseline.json').read_text(encoding='utf-8-sig'))
    baseline_model, _ = make_model('model', baseline_cfg, device)

    baseline_seconds, final_seconds = [], []
    for _ in range(2):
        result, rss = timed_score(baseline_model, validation, device, proc)
        baseline_seconds.append(result['seconds'])
        print(f"baseline  : {result['seconds']:6.2f}s  bpb={result['bpb']:.4f}  rss={rss/1e9:.2f}GB")
        result, rss = timed_score(final_model, validation, device, proc)
        final_seconds.append(result['seconds'])
        final_rss = rss
        print(f"final     : {result['seconds']:6.2f}s  bpb={result['bpb']:.4f}  rss={rss/1e9:.2f}GB")

    ratio = min(final_seconds) / max(baseline_seconds)
    asset_mb = args.checkpoint.stat().st_size / 1e6
    print('\n=== measured limits (same machine, 4 threads, fp32, validation) ===')
    print(f"CPU time ratio        : {ratio:.2f}x   (limit 5x)")
    print(f"peak process RAM      : {final_rss/1e9:.2f} GiB   (limit 4 GiB)")
    print(f"inference asset size  : {asset_mb:.1f} MB   (limit 64 MiB uncompressed)")


if __name__ == '__main__':
    main()
