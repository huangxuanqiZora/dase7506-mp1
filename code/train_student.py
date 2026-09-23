"""Improved training recipe for the MP1 student model.

Differences from the classroom ``train.py``:
    - configurable architecture config (defaults to ``configs/student.json``)
    - decoupled weight decay (no decay on norms / embeddings)
    - longer linear warmup and cosine decay to a configurable floor
    - optional exponential moving average (EMA) of the weights
    - periodic validation scoring for both raw and EMA weights
    - the checkpoint written as ``checkpoint.pt`` holds the state that scored
      best on validation, so the frozen predictor is selected without touching
      the test split.

Usage:
    python train_student.py --run-dir runs/student --steps 20000
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import (PROTOCOL, ROOT, autocast, device_metrics, load_data,
                    make_model, setup, sha)
from evaluate import score


class EMA:
    """Exponential moving average of floating point model parameters."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self):
        return self.shadow


def make_optimizer(model, lr, weight_decay, betas):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [{'params': decay, 'weight_decay': weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


def lr_at(step, steps, base_lr, min_ratio, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * coeff)


@torch.no_grad()
def evaluate_state(model, state_dict, data, device, split='validation'):
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state_dict)
    result = score(model, *data[split], device, 'fp32')
    result.pop('window_nll_nats', None)
    model.load_state_dict(backup)
    return result


def consider(best, source, step, state, bpb):
    """Keep the validation-best state seen so far (checkpoint selection / early stop)."""
    if bpb < best['bpb']:
        return {'bpb': bpb, 'step': step, 'source': source,
                'state': {k: v.detach().clone() for k, v in state.items()}}
    return best


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/student.json')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/student')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr-ratio', type=float, default=0.1)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--no-ema', action='store_true')
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--compile', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = make_optimizer(model, args.lr, args.weight_decay, (args.beta1, args.beta2))
    ema = None if args.no_ema else EMA(model, args.ema_decay)
    run_model = torch.compile(model) if args.compile else model
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()

    history, validation_history = [], []
    intermediate_validation_seconds = 0.0
    best = {'bpb': float('inf'), 'step': 0, 'source': None, 'state': None}
    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        lr = lr_at(step, args.steps, args.lr, args.min_lr_ratio, args.warmup)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(run_model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(), 'lr': lr,
                   'seconds': time.perf_counter() - started - intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            raw = evaluate_state(model, model.state_dict(), data, device)
            intermediate_validation_seconds += raw['seconds']
            entry = {'step': step + 1, 'raw': raw}
            best = consider(best, 'raw', step + 1, model.state_dict(), raw['bpb'])
            if ema is not None:
                ema_res = evaluate_state(model, ema.state_dict(), data, device)
                intermediate_validation_seconds += ema_res['seconds']
                entry['ema'] = ema_res
                best = consider(best, 'ema', step + 1, ema.state_dict(), ema_res['bpb'])
            validation_history.append(entry)
            print(json.dumps({'validation': entry}), flush=True)

    raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    candidates = {'raw': raw_state}
    if ema is not None:
        candidates['ema'] = ema.state_dict()
    for source, state in candidates.items():
        result = evaluate_state(model, state, data, device)
        intermediate_validation_seconds += result['seconds']
        best = consider(best, source, args.steps, state, result['bpb'])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds

    checkpoint = args.run_dir / 'checkpoint.pt'
    torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                'model': best['state'], 'seed': args.seed,
                'train_tokens': args.steps * args.batch_size * 256}, checkpoint)
    torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                'model': raw_state, 'seed': args.seed,
                'train_tokens': args.steps * args.batch_size * 256}, args.run_dir / 'checkpoint_raw.pt')
    if ema is not None:
        torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                    'model': ema.state_dict(), 'seed': args.seed,
                    'train_tokens': args.steps * args.batch_size * 256}, args.run_dir / 'checkpoint_ema.pt')

    result = {'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
              'seed': args.seed, 'parameters': sum(p.numel() for p in model.parameters()),
              'precision': precision, 'train_tokens': args.steps * args.batch_size * 256,
              'preparation_seconds': preparation_seconds, 'train_seconds': train_seconds,
              'selected': {'bpb': best['bpb'], 'source': best['source'], 'step': best['step']},
              'history': history, 'validation_history': validation_history,
              'intermediate_validation_seconds': intermediate_validation_seconds,
              'process_seconds': time.perf_counter() - total_started,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'checkpoint_sha256': sha(checkpoint), 'implementation_sha256': implementation_sha,
              **device_metrics(device)}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result | {'history': [], 'validation_history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()
