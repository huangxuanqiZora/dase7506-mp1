"""Run a batch of MP1 experiments described by a JSON spec.

Each experiment trains one model (provided ``train.py`` or our
``train_student.py``) into ``runs/<name>`` and records the validation BPB in a
summary table. Existing runs are skipped unless ``--force`` is given, so the
suite can be resumed across GPU sessions.

Spec format::

    {"experiments": [
        {"name": "student_eq", "trainer": "student", "implementation": "student",
         "config": "configs/student.json", "config_overrides": {"depth": 6},
         "steps": 1200, "batch_size": 32, "lr": 0.001, "warmup": 100,
         "min_lr_ratio": 0.1, "weight_decay": 0.1, "ema_decay": 0.999,
         "eval_every": 300, "seed": 17}
    ]}

Usage:
    python suite.py --spec experiments.json --device cuda
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
GENERATED = ROOT / 'configs/generated'


def build_config(exp):
    base = json.loads((ROOT / exp.get('config', 'configs/student.json')).read_text(encoding='utf-8-sig'))
    base.update(exp.get('config_overrides', {}))
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{exp['name']}.json"
    path.write_text(json.dumps(base, indent=2) + '\n')
    return path, base


def build_argv(exp, config_path, device, threads, runs_dir):
    if exp.get('trainer', 'student') == 'provided':
        argv = [sys.executable, 'train.py',
                '--implementation', exp.get('implementation', 'model'),
                '--config', str(config_path),
                '--run-dir', str(runs_dir / exp['name']),
                '--device', device, '--threads', str(threads),
                '--seed', str(exp.get('seed', 17)),
                '--steps', str(exp['steps']),
                '--batch-size', str(exp.get('batch_size', 32))]
        if exp.get('eval_every', 0):
            argv += ['--eval-every', str(exp['eval_every'])]
        return argv
    argv = [sys.executable, 'train_student.py',
            '--implementation', exp.get('implementation', 'student'),
            '--config', str(config_path),
            '--run-dir', str(runs_dir / exp['name']),
            '--device', device, '--threads', str(threads),
            '--seed', str(exp.get('seed', 17)),
            '--steps', str(exp['steps']),
            '--batch-size', str(exp.get('batch_size', 32)),
            '--lr', str(exp.get('lr', 1e-3)),
            '--min-lr-ratio', str(exp.get('min_lr_ratio', 0.1)),
            '--warmup', str(exp.get('warmup', 100)),
            '--weight-decay', str(exp.get('weight_decay', 0.1)),
            '--beta1', str(exp.get('beta1', 0.9)),
            '--beta2', str(exp.get('beta2', 0.95)),
            '--grad-clip', str(exp.get('grad_clip', 1.0)),
            '--ema-decay', str(exp.get('ema_decay', 0.999)),
            '--eval-every', str(exp.get('eval_every', 500)),
            '--log-every', str(exp.get('log_every', 100))]
    if exp.get('no_ema'):
        argv += ['--no-ema']
    if exp.get('compile'):
        argv += ['--compile']
    return argv


def summarize(exp, metrics):
    result = {'name': exp['name'], 'parameters': metrics.get('parameters'),
              'train_tokens': metrics.get('train_tokens'),
              'train_seconds': metrics.get('train_seconds'),
              'seed': metrics.get('seed')}
    if 'selected' in metrics:
        result['validation_bpb'] = metrics['selected']['bpb']
        result['selected_source'] = metrics['selected']['source']
    else:
        result['validation_bpb'] = metrics['validation']['bpb']
        result['selected_source'] = 'raw'
    return result


def write_summary(runs_dir, results):
    """Persist the summary after every experiment so a reset cannot lose it."""
    (runs_dir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    with (runs_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['name', 'validation_bpb', 'parameters',
                                                    'train_tokens', 'train_seconds', 'seed',
                                                    'selected_source'])
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--only', default='', help='Comma separated experiment names.')
    p.add_argument('--runs-dir', type=Path, default=RUNS,
                   help='Where to store runs (keep it outside the code directory).')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    experiments = json.loads(args.spec.read_text(encoding='utf-8-sig'))['experiments']
    wanted = {n.strip() for n in args.only.split(',') if n.strip()}
    if wanted:
        experiments = [e for e in experiments if e['name'] in wanted]
    runs_dir = args.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for exp in experiments:
        run_dir = runs_dir / exp['name']
        metrics_path = run_dir / 'metrics.json'
        if metrics_path.exists() and not args.force:
            metrics = json.loads(metrics_path.read_text())
            row = summarize(exp, metrics)
            results.append(row)
            print(f"[skip] {exp['name']}: validation_bpb={row['validation_bpb']:.4f}", flush=True)
            continue
        config_path, config = build_config(exp)
        argv = build_argv(exp, config_path, args.device, args.threads, runs_dir)
        print(f"[run ] {exp['name']}: {' '.join(argv[1:])}", flush=True)
        log_path = runs_dir / f"{exp['name']}.log"
        with log_path.open('w') as log:
            proc = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
            log.write(proc.stdout)
        if not metrics_path.exists():
            print(f"[fail] {exp['name']} produced no metrics.json (exit {proc.returncode}); see {log_path}", flush=True)
            results.append({'name': exp['name'], 'validation_bpb': None, 'error': proc.returncode})
            continue
        metrics = json.loads(metrics_path.read_text())
        row = summarize(exp, metrics)
        results.append(row)
        print(f"[done] {exp['name']}: validation_bpb={row['validation_bpb']:.4f} "
              f"params={row['parameters']} time={row['train_seconds']:.0f}s", flush=True)
        write_summary(runs_dir, results)

    write_summary(runs_dir, results)
    print('\n=== summary (lower BPB is better) ===', flush=True)
    for row in sorted(results, key=lambda r: (r.get('validation_bpb') is None, r.get('validation_bpb', 0))):
        bpb = row.get('validation_bpb')
        print(f"{row['name']:<28} {bpb if bpb is None else round(bpb, 4)}", flush=True)


if __name__ == '__main__':
    main()
