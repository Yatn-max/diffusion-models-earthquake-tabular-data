import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import pandas as pd
import tomli
import tomli_w
from utils import (
    TARGET, run_cmd, rm_paths,
    strip_verbose, prep_seismic_df, fix_output_cols
)


def main():
    parser = argparse.ArgumentParser(description="Train and sample TabRep on seismic data")
    parser.add_argument("--input", required=True, help="Path to real training CSV")
    parser.add_argument("--output", default="data/synthetic/tabrep_synthetic.csv", help="Output synthetic CSV path")
    parser.add_argument("--name", default="seismic_eew", help="Internal dataset name")
    parser.add_argument("--n_syn", type=int, default=10000, help="Number of synthetic rows to generate")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--work_dir", default="./external_repos", help="Directory to clone TabRep")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    py = sys.executable
    tr_dir = os.path.abspath(os.path.join(args.work_dir, "TabRep"))

    # 1. Load & prepare data
    df = prep_seismic_df(pd.read_csv(args.input), "train data")
    df["dummy_cat"] = np.random.RandomState(0).choice(["a", "b"], size=len(df))
    cols = list(df.columns)
    tidx = cols.index(TARGET)
    info = {
        "name": args.name,
        "task_type": "regression",
        "header": "infer",
        "column_names": None,
        "num_col_idx": [i for i, c in enumerate(cols) if i != tidx and c != "dummy_cat"],
        "cat_col_idx": [cols.index("dummy_cat")],
        "target_col_idx": [tidx],
        "file_type": "csv",
        "data_path": f"data/{args.name}/{args.name}.csv",
        "test_path": None
    }

    # 2. Clone & setup TabRep repository
    if not os.path.exists(tr_dir):
        run_cmd("git clone https://github.com/jacobyhsi/TabRep.git", cwd=args.work_dir)
    strip_verbose(tr_dir)

    os.makedirs(os.path.join(tr_dir, "data", args.name), exist_ok=True)
    os.makedirs(os.path.join(tr_dir, "data", "Info"), exist_ok=True)
    df.to_csv(os.path.join(tr_dir, "data", args.name, f"{args.name}.csv"), index=False)
    with open(os.path.join(tr_dir, "data", "Info", f"{args.name}.json"), "w") as f:
        json.dump(info, f, indent=4)

    for p in glob.glob(os.path.join(tr_dir, f"data/{args.name}/*.npy")):
        rm_paths(p)
    rm_paths(os.path.join(tr_dir, f"synthetic/{args.name}"), os.path.join(tr_dir, f"tabrep_flow/ckpt/{args.name}"))

    d_flag = f" --dataname {args.name}"
    run_cmd(f"{py} process_dataset.py{d_flag}", cwd=tr_dir)
    d_num = np.load(os.path.join(tr_dir, f"data/{args.name}/X_num_train.npy"), allow_pickle=True).shape[1]

    # 3. Build TOML config from template
    with open(os.path.join(tr_dir, "configs/beijing.toml"), "rb") as f:
        cfg = tomli.load(f)

    def sub(o):
        if isinstance(o, dict):
            return {k: sub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sub(v) for v in o]
        if isinstance(o, str):
            return o.replace("beijing", args.name)
        return o

    cfg = sub(cfg)
    cfg["num_numerical_features"] = d_num
    cfg["task_type"] = "regression"
    cfg.setdefault("train", {}).setdefault("main", {})
    cfg["train"]["main"]["steps"] = args.steps
    cfg["train"]["main"]["batch_size"] = args.batch_size
    cfg.setdefault("sample", {"batch_size": 10000})
    cfg["sample"]["num_samples"] = args.n_syn

    cfg_path = os.path.join(tr_dir, f"configs/{args.name}.toml")
    with open(cfg_path, "wb") as f:
        tomli_w.dump(cfg, f)

    # 4. Train & sample
    t0 = time.time()
    run_cmd(f"{py} main.py{d_flag} --method tabrep_flow --mode train", cwd=tr_dir)
    run_cmd(f"{py} main.py{d_flag} --method tabrep_flow --mode sample", cwd=tr_dir)

    outs = [
        p for p in glob.glob(os.path.join(tr_dir, "**", "*.csv"), recursive=True)
        if os.path.getmtime(p) > t0 and "/data/" not in p.replace("\\", "/")
    ]
    if not outs:
        raise RuntimeError("TabRep produced no synthetic CSV output.")

    src = sorted(outs, key=os.path.getmtime)[-1]
    s = fix_output_cols(pd.read_csv(src))
    if len(s) != args.n_syn:
        s = s.sample(n=args.n_syn, replace=len(s) < args.n_syn, random_state=42)
    s = s.reset_index(drop=True)
    s.to_csv(args.output, index=False)
    print(f"Saved {args.output} {s.shape} | Synthetic Magnitude: {s[TARGET].min():.2f} - {s[TARGET].max():.2f}")


if __name__ == "__main__":
    main()