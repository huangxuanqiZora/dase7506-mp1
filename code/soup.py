"""Average the weights of several checkpoints (model soup) and score them.

Weight averaging is allowed by the guide ("self-trained weight averaging") and
costs nothing at inference time: the soup is a single predictor with the same
architecture as its members. Members must share one config.

Usage:
    python soup.py --checkpoints runs/final_s17/checkpoint.pt runs/final_s42/checkpoint.pt \
        --output runs/soup/checkpoint.pt --split validation
"""
import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, ROOT, make_model, setup, load_data, sha
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--split', default='validation')
    args = p.parse_args()

    device, precision = setup(args.device, 'fp32', args.threads)
    states = [torch.load(c, map_location='cpu', weights_only=True) for c in args.checkpoints]
    config = states[0]['config']
    for c, state in zip(args.checkpoints, states):
        if state['config'] != config:
            raise ValueError(f'{c} has a different config; soups need identical configs.')
    keys = states[0]['model'].keys()
    soup = {}
    for key in keys:
        tensors = [s['model'][key] for s in states]
        soup[key] = torch.stack(tensors).mean(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'protocol': PROTOCOL, 'implementation': states[0]['implementation'],
                'config': config, 'model': soup,
                'members': [str(c) for c in args.checkpoints]}, args.output)

    data = load_data()
    model, _ = make_model(states[0]['implementation'], config, device)
    model.load_state_dict(soup)
    result = score(model, *data[args.split], device, 'fp32')
    result.pop('window_nll_nats', None)
    result.update(split=args.split, members=[str(c) for c in args.checkpoints],
                  parameters=sum(p.numel() for p in model.parameters()),
                  checkpoint_sha256=sha(args.output))
    (args.output.parent / f"{args.split}_soup.json").write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
