import os
import re
import glob
import shutil
import subprocess
import pandas as pd

WANT = [
    "Tau_c", "PI_v", "T_va", "RSSC Vel", "CAV", "CAD",
    "ID_2", "IV_2", "Arias intensity UD", "PGV UD 3 sec",
    "PGD UD 3 sec", "Magnitude"
]
TARGET = "Magnitude"


def run_cmd(cmd, cwd=None):
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(
        cmd, shell=True, cwd=cwd, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for line in p.stdout:
        print(line, end="")
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {cmd}")
    return True


def rm_paths(*paths):
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.exists(p):
            os.remove(p)


def patch_file(path, pattern, repl, label):
    with open(path, "r") as f:
        t = f.read()
    t2 = re.sub(pattern, repl, t)
    with open(path, "w") as f:
        f.write(t2)
    print(label, "-> patched" if t != t2 else "-> unchanged/not found")


def strip_verbose(repo_dir):
    for fp in glob.glob(os.path.join(repo_dir, "**", "*.py"), recursive=True):
        with open(fp, "r") as f:
            t = f.read()
        if ", verbose=True)" in t:
            with open(fp, "w") as f:
                f.write(t.replace(", verbose=True)", ")"))


def prep_seismic_df(raw: pd.DataFrame, label: str = "dataset", impute: bool = True) -> pd.DataFrame:
    look = {re.sub(r"\s+", " ", c.strip()).lower(): c for c in raw.columns}
    miss = [w for w in WANT if w.lower() not in look]
    if miss:
        raise ValueError(f"{label} is missing columns: {miss}\nAvailable: {list(raw.columns)}")
    d = raw[[look[w.lower()] for w in WANT]].copy()
    d.columns = WANT
    d = d.apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=[TARGET])
    d = d.fillna(d.median()) if impute else d.dropna()
    return d.reset_index(drop=True)


def fix_output_cols(s: pd.DataFrame) -> pd.DataFrame:
    if not all(c in s.columns for c in WANT):
        if s.shape[1] == len(WANT) + 1:
            print("Output columns are not named; assuming original order.")
            s = s.copy()
            s.columns = WANT + ["dummy_cat"]
        else:
            raise ValueError(f"Unexpected output columns: {list(s.columns)}")
    return s[WANT]