#!/usr/bin/env python3
"""
Analysis script for the GA hyperparameter sweep results.

Reads results/runs.csv and results/generations.csv and saves report-ready
PNG plots to results/.

Usage:
    python3 analysis.py
"""

"""
I am not happy with the summary results right now but I think it's because
there isn't enough data right now - Tyler
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

RESULTS_DIR = "results"
RUNS_CSV    = os.path.join(RESULTS_DIR, "runs.csv")
GENS_CSV    = os.path.join(RESULTS_DIR, "generations.csv")

BASELINE = {
    "pop_size":        100,
    "tournament_size":   5,
    "crossover_rate":  0.7,
    "mutation_rate":   0.1,
    "mutation_sigma":  0.1,
    "elitism_count":     5,
}

PARAM_LABELS = {
    "pop_size":        "Population Size",
    "tournament_size": "Tournament Size",
    "crossover_rate":  "Crossover Rate",
    "mutation_rate":   "Mutation Rate",
    "mutation_sigma":  "Mutation Sigma",
    "elitism_count":   "Elitism Count",
}

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def sweep_mask(runs: pd.DataFrame, vary_param: str) -> pd.Series:
    """Rows where every param except vary_param is at its baseline value."""
    mask = pd.Series(True, index=runs.index)
    for p, v in BASELINE.items():
        if p != vary_param:
            mask &= runs[p].round(9) == round(v, 9)
    return mask


# ── Convergence curves ────────────────────────────────────────────

def plot_convergence(runs: pd.DataFrame, gens: pd.DataFrame,
                     vary_param: str, metric: str,
                     ylabel: str, title_prefix: str, out_path: str):
    valid_ids = runs.loc[sweep_mask(runs, vary_param), "run_id"]
    df = gens[gens["run_id"].isin(valid_ids)].copy()
    df = df.merge(runs[["run_id", vary_param]], on="run_id")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, val in enumerate(sorted(df[vary_param].unique())):
        group = df[df[vary_param] == val].groupby("generation")[metric]
        means = group.mean()
        stds  = group.std().fillna(0)
        scale = 100 if metric == "landing_rate" else 1
        color = COLORS[i % len(COLORS)]
        ax.plot(means.index, means * scale, color=color,
                label=f"{PARAM_LABELS[vary_param]} = {val}", linewidth=1.5)
        ax.fill_between(means.index,
                        (means - stds) * scale, (means + stds) * scale,
                        color=color, alpha=0.15)

    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — varying {PARAM_LABELS[vary_param]}")
    if metric == "landing_rate":
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Dot plots with trial scatter (2×3 grid) ──────────────────────

def plot_dot_grid(runs: pd.DataFrame, metric: str, ylabel: str,
                  title: str, out_path: str, pct: bool = False):
    params = list(BASELINE.keys())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    col = runs[metric].copy()
    if metric == "converged_gen":
        never = col == -1
        col[never] = col[~never].max() + 1 if (~never).any() else 301

    for ax_i, vary_param in enumerate(params):
        ax = axes[ax_i]
        subset = runs[sweep_mask(runs, vary_param)].copy()
        subset = subset.assign(**{metric: col[subset.index]})

        vals = sorted(subset[vary_param].unique())
        scale = 100 if pct else 1
        xs = range(len(vals))

        for x, val in zip(xs, vals):
            points = subset[subset[vary_param] == val][metric].values * scale
            mean   = points.mean()
            # individual trial dots with jitter
            jitter = (pd.Series(range(len(points))) - (len(points) - 1) / 2) * 0.08
            ax.scatter(x + jitter, points, color=COLORS[x % len(COLORS)],
                       s=40, zorder=3, alpha=0.7)
            # mean line spanning ±0.25 around the x position
            ax.plot([x - 0.25, x + 0.25], [mean, mean],
                    color=COLORS[x % len(COLORS)], linewidth=2.5, zorder=4)

            # mark baseline with a dashed vertical line
            if round(float(val), 9) == round(BASELINE[vary_param], 9):
                ax.axvline(x, color="black", linewidth=1, linestyle="--", alpha=0.4)

        ax.set_xticks(list(xs))
        ax.set_xticklabels([str(v) for v in vals])
        ax.set_xlabel(PARAM_LABELS[vary_param], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(PARAM_LABELS[vary_param], fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if pct:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RUNS_CSV) or not os.path.exists(GENS_CSV):
        print(f"Error: CSV files not found in {RESULTS_DIR}/")
        print("Run 'python3 experiment.py' first.")
        return

    print("Loading CSVs...")
    runs = pd.read_csv(RUNS_CSV)
    gens = pd.read_csv(GENS_CSV)
    print(f"  {len(runs)} runs,  {len(gens)} generation rows\n")

    print("Generating convergence curves...")
    for p in BASELINE:
        plot_convergence(runs, gens, p,
                         "best_fitness", "Best Fitness", "Convergence",
                         os.path.join(RESULTS_DIR, f"convergence_fitness_{p}.png"))
        plot_convergence(runs, gens, p,
                         "landing_rate", "Landing Rate (%)", "Landing Rate",
                         os.path.join(RESULTS_DIR, f"convergence_landing_{p}.png"))

    print("\nGenerating summary dot plots...")
    plot_dot_grid(runs, "final_best_fitness", "Final Best Fitness",
                  "Final Best Fitness by Hyperparameter",
                  os.path.join(RESULTS_DIR, "summary_final_fitness.png"))

    plot_dot_grid(runs, "final_landing_rate", "Final Landing Rate (%)",
                  "Final Landing Rate by Hyperparameter",
                  os.path.join(RESULTS_DIR, "summary_landing_rate.png"), pct=True)

    plot_dot_grid(runs, "converged_gen", "Generation",
                  "Convergence Speed (first gen ≥ 50% landing rate)\n"
                  "— point at max means never converged —",
                  os.path.join(RESULTS_DIR, "summary_convergence_speed.png"))

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
