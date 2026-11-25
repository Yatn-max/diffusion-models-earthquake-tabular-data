import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from utils import (
    TARGET, run_cmd, rm_paths, patch_file,
    strip_verbose, prep_seismic_df, fix_output_cols
)


def main():
    parser = argparse.ArgumentParser(description="Train and sample TabSyn on seismic data")
    parser.add_argument("--input", required=True, help="Path to real training CSV")
    parser.add_argument("--output", default="data/synthetic/tabsyn_synthetic.csv", help="Output synthetic CSV path")
    parser.add_argument("--name", default="seismic_eew", help="Internal dataset name")
    parser.add_argument("--n_syn", type=int, default=10000, help="Number of synthetic rows to generate")
    parser.add_argument("--vae_epochs", type=int, default=500, help="VAE training epochs")
    parser.add_argument("--diff_epochs", type=int, default=1500, help="Diffusion training epochs")
    parser.add_argument("--work_dir", default="./external_repos", help="Directory to clone TabSyn")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    py = sys.executable
    ts_dir = os.path.abspath(os.path.join(args.work_dir, "tabsyn"))

    # 1. Load & prepare data
    df = prep_seismic_df(pd.read_csv(args.input), "train data")
    print(f"Loaded {args.input}: {df.shape} | Magnitude range: {df[TARGET].min():.2f} - {df[TARGET].max():.2f}")

    # TabSyn requires at least one categorical column
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

    # 2. Clone & patch TabSyn repository
    if not os.path.exists(ts_dir):
        run_cmd("git clone https://github.com/amazon-science/tabsyn.git", cwd=args.work_dir)

    os.makedirs(os.path.join(ts_dir, "data", args.name), exist_ok=True)
    os.makedirs(os.path.join(ts_dir, "data", "Info"), exist_ok=True)
    df.to_csv(os.path.join(ts_dir, "data", args.name, f"{args.name}.csv"), index=False)
    with open(os.path.join(ts_dir, "data", "Info", f"{args.name}.json"), "w") as f:
        json.dump(info, f, indent=4)

    strip_verbose(ts_dir)
    patch_file(os.path.join(ts_dir, "tabsyn/vae/main.py"), r"NUM_EPOCHS\s*=\s*\d+", f"NUM_EPOCHS = {args.vae_epochs}", "VAE epochs")
    patch_file(os.path.join(ts_dir, "tabsyn/main.py"), r"num_epochs\s*=\s*\d+(\s*\+\s*\d+)?", f"num_epochs = {args.diff_epochs}", "Diffusion epochs")
    patch_file(os.path.join(ts_dir, "tabsyn/sample.py"), r"num_samples\s*=\s*[\w\.\[\]]+", f"num_samples = {args.n_syn}", "Sample size")

    # 3. Clean stale artifacts & run pipeline
    for p in glob.glob(os.path.join(ts_dir, f"data/{args.name}/*.npy")):
        rm_paths(p)
    rm_paths(
        os.path.join(ts_dir, f"tabsyn/vae/ckpt/{args.name}"),
        os.path.join(ts_dir, f"tabsyn/ckpt/{args.name}"),
        os.path.join(ts_dir, f"synthetic/{args.name}")
    )

    d_flag = f" --dataname {args.name}"
    run_cmd(f"{py} process_dataset.py{d_flag}", cwd=ts_dir)
    run_cmd(f"{py} main.py{d_flag} --method vae --mode train", cwd=ts_dir)
    run_cmd(f"{py} main.py{d_flag} --method tabsyn --mode train", cwd=ts_dir)
    run_cmd(f"{py} main.py{d_flag} --method tabsyn --mode sample", cwd=ts_dir)

    # 4. Save clean output
    out_csv = os.path.join(ts_dir, f"synthetic/{args.name}/tabsyn.csv")
    s = fix_output_cols(pd.read_csv(out_csv))
    if len(s) != args.n_syn:
        s = s.sample(n=args.n_syn, replace=len(s) < args.n_syn, random_state=42)
    s = s.reset_index(drop=True)
    s.to_csv(args.output, index=False)
    print(f"Saved {args.output} {s.shape} | Synthetic Magnitude: {s[TARGET].min():.2f} - {s[TARGET].max():.2f}")


if __name__ == "__main__":
    main()