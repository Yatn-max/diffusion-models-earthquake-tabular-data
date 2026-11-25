import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import WANT, TARGET, prep_seismic_df


def plot_heatmap(ax, mat, title, vmin, vmax, cmap):
    im = ax.imshow(mat.values, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(WANT)))
    ax.set_yticks(range(len(WANT)))
    ax.set_xticklabels(WANT, rotation=90, fontsize=8)
    ax.set_yticklabels(WANT, fontsize=8)
    for i in range(len(WANT)):
        for j in range(len(WANT)):
            v = mat.values[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(v) > 0.6 * max(abs(vmin), abs(vmax)) else "black"
            )
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Real vs TabSyn vs TabRep correlation fidelity")
    parser.add_argument("--real", required=True, help="Path to real seismic CSV")
    parser.add_argument("--tabsyn", default=None, help="Path to TabSyn synthetic CSV")
    parser.add_argument("--tabrep", default=None, help="Path to TabRep synthetic CSV")
    parser.add_argument("--assets_dir", default="assets", help="Directory to save output PNG charts")
    args = parser.parse_args()

    os.makedirs(args.assets_dir, exist_ok=True)
    sets = {"Real": prep_seismic_df(pd.read_csv(args.real), "Real", impute=False)}

    if args.tabsyn and os.path.exists(args.tabsyn):
        sets["TabSyn"] = prep_seismic_df(pd.read_csv(args.tabsyn), "TabSyn", impute=False)
    if args.tabrep and os.path.exists(args.tabrep):
        sets["TabRep"] = prep_seismic_df(pd.read_csv(args.tabrep), "TabRep", impute=False)

    if len(sets) < 2:
        raise ValueError("Provide at least one synthetic dataset via --tabsyn or --tabrep.")

    corr = {k: v.corr(method="pearson") for k, v in sets.items()}
    syn_names = [k for k in sets if k != "Real"]
    iu = np.triu_indices(len(WANT), k=1)

    # 1. Side-by-side Correlation Heatmaps
    fig, axes = plt.subplots(1, len(sets), figsize=(9 * len(sets), 8.5))
    axes = np.atleast_1d(axes)
    for ax, (k, mat) in zip(axes, corr.items()):
        plot_heatmap(ax, mat, f"Pearson correlation: {k}", -1, 1, "coolwarm")
    plt.tight_layout()
    plt.savefig(os.path.join(args.assets_dir, "p1_correlations.png"), dpi=150)
    plt.close()

    # 2. Absolute Correlation Difference vs Real
    fig, axes = plt.subplots(1, len(syn_names), figsize=(9 * len(syn_names), 8.5))
    axes = np.atleast_1d(axes)
    print("\nMean absolute difference in correlations vs Real (lower = better):")
    for ax, k in zip(axes, syn_names):
        diff = (corr[k] - corr["Real"]).abs()
        mad = diff.values[iu].mean()
        print(f"  {k:<8} MAD: {mad:.4f} (max single-pair gap: {diff.values[iu].max():.4f})")
        plot_heatmap(ax, diff, f"|{k} - Real|   mean = {mad:.4f}", 0, 0.5, "Reds")
    plt.tight_layout()
    plt.savefig(os.path.join(args.assets_dir, "p2_difference.png"), dpi=150)
    plt.close()

    # 3. Feature vs Magnitude Correlation Bar Chart
    feats = [c for c in WANT if c != TARGET]
    x = np.arange(len(feats))
    w = 0.8 / len(sets)
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (k, mat) in enumerate(corr.items()):
        bars = ax.bar(x + i * w - 0.4 + w / 2, mat.loc[feats, TARGET].values, w, label=k)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, rotation=90, padding=2)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=35, ha="right")
    ax.set_ylabel(f"Pearson r with {TARGET}")
    ax.set_title("Seismic P-Wave Feature vs. Magnitude Correlation")
    ax.margins(y=0.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.assets_dir, "p3_magnitude_corr.png"), dpi=150)
    plt.close()
    print(f"\nSaved plots to {args.assets_dir}/")


if __name__ == "__main__":
    main()