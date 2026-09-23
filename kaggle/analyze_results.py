import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
names = sys.argv[2:] or [p.name for p in base.iterdir() if p.is_dir()]
for name in names:
    metrics_path = base / name / 'metrics.json'
    if not metrics_path.exists():
        continue
    m = json.loads(metrics_path.read_text())
    print(f"== {name}  params={m['parameters']}  selected={m['selected']}")
    for entry in m.get('validation_history', []):
        line = f"   step {entry['step']:>6}: raw {entry['raw']['bpb']:.4f}"
        if 'ema' in entry:
            line += f"  ema {entry['ema']['bpb']:.4f}"
        print(line)
