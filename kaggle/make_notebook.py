"""Generate the Kaggle notebook that runs the MP1 experiments.

The notebook embeds the current ``student.py``, ``train_student.py`` and
``suite.py`` so the Kaggle dataset only has to hold the immutable supplied data
and evaluator. Regenerate this notebook whenever the code changes.

    python make_notebook.py
"""
import json
import pathlib

CODE = pathlib.Path(__file__).resolve().parent.parent / 'code'
OUT = pathlib.Path(__file__).resolve().parent / 'MP1_Kaggle.ipynb'


def read(name):
    return (CODE / name).read_text(encoding='utf-8')


SETUP = r'''import os, sys, glob, shutil, subprocess, pathlib, json, time
started = time.time()

hits = glob.glob('/kaggle/input/**/data/tokenizer.json', recursive=True)
assert hits, 'No dataset containing code/data/tokenizer.json was found under /kaggle/input.'
src_code = pathlib.Path(hits[0]).parent.parent
print('dataset code dir:', src_code)

dst_code = pathlib.Path('/kaggle/working/code')
if dst_code.exists():
    shutil.rmtree(dst_code)
shutil.copytree(src_code, dst_code)
os.chdir(dst_code)
print('working dir:', os.getcwd())

try:
    import tokenizers  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers==0.21.4'], check=True)

import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported())
    print('memory GB:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
'''

SMOKE = [
    {"name": "baseline_repro", "trainer": "provided", "implementation": "model",
     "config": "configs/baseline.json", "steps": 1200, "batch_size": 32,
     "eval_every": 0, "seed": 17},
    {"name": "student_smoke", "trainer": "student", "implementation": "student",
     "config": "configs/student.json", "steps": 300, "batch_size": 32,
     "lr": 0.001, "warmup": 50, "min_lr_ratio": 0.1, "eval_every": 300, "seed": 17},
]

FULL = [
    # --- already completed before the session reset (kept here as a record) ---
    # eq_baseline_arch 2.0795 | eq_student_arch 1.6970 | long_8k 1.5560
    # abl_no_rope 1.5593 | abl_gelu 1.5271 | abl_layernorm 1.5583
    # Final configuration (GELU, depth 8, dropout 0.2, 20k steps, val selection),
    # trained with extra seeds so the checkpoints can be averaged into a soup.
    {"name": "final_s42", "trainer": "student", "implementation": "student",
     "config": "configs/student.json",
     "config_overrides": {"mlp": "gelu", "depth": 8, "dropout": 0.2},
     "steps": 20000, "batch_size": 32, "lr": 0.001, "warmup": 500,
     "min_lr_ratio": 0.1, "eval_every": 2000, "seed": 42},
    {"name": "final_s123", "trainer": "student", "implementation": "student",
     "config": "configs/student.json",
     "config_overrides": {"mlp": "gelu", "depth": 8, "dropout": 0.2},
     "steps": 20000, "batch_size": 32, "lr": 0.001, "warmup": 500,
     "min_lr_ratio": 0.1, "eval_every": 2000, "seed": 123},
]

STUDENT_CONFIG = read('configs/student.json')


def code_cell(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


cells = [
    md_cell("""# DASE7506 MP1 - Small Language Model Challenge (Kaggle runner)

This notebook trains and evaluates the improved model on the supplied WikiText-2
data. Before running:

1. Create a Kaggle **Dataset** from `mp1_data.zip` (it extracts to `code/...`).
2. Create a new Notebook, **Add Input -> your dataset**, and select a GPU
   (Settings -> Accelerator -> GPU T4 x2 or P100).
3. Set `MODE` in the *Experiment list* cell to `"smoke"` for a first check, then
   `"full"` for the complete suite.
4. Run all cells. Results are printed at the end and saved to
   `/kaggle/working/mp1_results.zip` for download.

The supplied data and evaluator are never modified; only `student.py`,
`train_student.py`, `suite.py` and the config are written here.
"""),
    code_cell(SETUP),
    code_cell('%%writefile /kaggle/working/code/student.py\n' + read('student.py')),
    code_cell('%%writefile /kaggle/working/code/train_student.py\n' + read('train_student.py')),
    code_cell('%%writefile /kaggle/working/code/suite.py\n' + read('suite.py')),
    code_cell(
        'MODE = "full"   # "smoke" = quick check, "full" = experiment suite\n\n'
        'import json, pathlib\n'
        'SMOKE = %s\n'
        'FULL = %s\n'
        'pathlib.Path("configs").mkdir(exist_ok=True)\n'
        'pathlib.Path("configs/student.json").write_text(%r)\n\n'
        'experiments = {"smoke": SMOKE, "full": FULL}[MODE]\n'
        'pathlib.Path("experiments.json").write_text(json.dumps({"experiments": experiments}, indent=2))\n'
        'print("MODE =", MODE, "| experiments:", [e["name"] for e in experiments])\n'
        % (repr(SMOKE), repr(FULL), STUDENT_CONFIG)),
    code_cell('!python suite.py --spec experiments.json --device cuda --threads 4 --runs-dir /kaggle/working/runs'),
    code_cell(r'''import shutil, pathlib
runs = pathlib.Path('/kaggle/working/runs')
print((runs / 'summary.csv').read_text())
shutil.make_archive('/kaggle/working/mp1_results', 'zip', str(runs))
for p in sorted(pathlib.Path('/kaggle/working').glob('mp1_results.zip')):
    print('saved', p, round(p.stat().st_size/1e6, 1), 'MB')'''),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print('wrote', OUT)
