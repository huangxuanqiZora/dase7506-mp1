"""Measure the scoring-time ratio of a checkpoint against the baseline on CPU."""
import json
import sys
import time
from pathlib import Path

import torch

from common import load_data, make_model, setup
from evaluate import score

device, precision = setup('cpu', 'fp32', 4)
data = load_data()
val = data['validation']


def time_model(implementation, config):
    model, _ = make_model(implementation, config, device)
    torch.set_num_threads(4)
    result = score(model, *val, device, 'fp32')
    return result['seconds'], result['bpb']


baseline_cfg = json.loads((Path('configs/baseline.json')).read_text())
base_seconds, base_bpb = time_model('model', baseline_cfg)
print(f"baseline   : {base_seconds:6.2f}s  bpb={base_bpb:.4f}")

for path in sys.argv[1:]:
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    seconds, bpb = time_model(ckpt['implementation'], ckpt['config'])
    print(f"{Path(path).parent.name:12s}: {seconds:6.2f}s  bpb={bpb:.4f}  ratio={seconds/base_seconds:.2f}x")
