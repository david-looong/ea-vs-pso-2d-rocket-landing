#!/usr/bin/env python3
"""
Showcase the project in a few graphs and a small set of headline numbers.

Two modes, controlled by the ``SWEEP`` constant near the top of the file:

  1. **Single-condition mode** (default; ``SWEEP["variable"] is None``):
     Train the *baseline* GA configuration (from ``config.py``) ``N`` times
     with deterministic seeds 1..N, plot fitness/landing curves with mean ±
     std bands across replicates, then pick a champion controller from each
     replicate (best mean fitness on a fixed 50-scenario validation set
     drawn from the last 10 generations) and stress-test it on a fixed
     1000-scenario robustness set. Output filenames stay identical to the
     pre-sweep version of this script, with one new pair of robustness
     histograms added.

  2. **Multi-condition sweep mode** (``SWEEP["variable"]`` set):
     Override one ``config.py`` constant per value in ``SWEEP["values"]``,
     re-run the entire single-condition pipeline for each value, then add
     a set of *combined* cross-condition charts that overlay the per-
     condition curves and stack the robustness CDFs.

The validation and robustness scenario sets (50 + 1000 fixed integer seeds)
are module-level constants — identical across every replicate, condition and
script invocation, so champion comparisons are apples-to-apples. Each seed
deterministically produces one set of random initial conditions sampled from
the same config ranges the GA uses for ``NUM_EVAL_TRIALS``.

Outputs (under ``results/<run-dir>/``):

Each invocation creates its own run folder under ``results/`` so nothing is
ever overwritten. The folder name encodes the mode and a timestamp, e.g.::

    results/showcase_baseline_2026-05-09_19-04-23/
    results/showcase_sweep_NN_LAYERS_2026-05-09_19-04-23/

Inside each run folder::

    showcase_data.json                     raw per-trial per-generation metrics
                                           + per-replicate champion robustness
    showcase_headline.json                 list of headline dicts (one/condition)
    dark/                                  all dark-theme PNGs for this run
        showcase_fitness_dark[_<cond>].png best & mean fitness vs generation
        showcase_landing_dark[_<cond>].png landing rate vs generation
        showcase_robustness_dark[_<cond>].png stacked-bar histogram of champion
                                             fitness on the 1000-seed set,
                                             pooled across replicates
        showcase_combined_*_dark.png       sweep-mode-only overlays
    light/                                 same set of plots, light theme

Usage:

    python showcase.py                          # train N seeds, plot, print
    python showcase.py --trials 7               # different number of seeds
    python showcase.py --run-dir results/foo    # force a specific output dir
    python showcase.py --plot-only              # re-plot from most-recent run
    python showcase.py --plot-only \
        --run-dir results/showcase_baseline_... # re-plot from a specific run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config
from controller import NeuralNetwork
import main as ga_main


# ═══════════════════════════════════════════════════════════════════════
#  User-tunable constants
# ═══════════════════════════════════════════════════════════════════════

# Sweep one ``config.py`` constant across multiple values, running the full
# pipeline (replicates × GA × champion selection × robustness) for each.
# Set ``"variable": None`` to disable sweeping and run the baseline only.
#
# Sweepable variables fall into two groups:
#   • Threaded through ``train()``'s arguments — these always work:
#       NN_LAYERS, POPULATION_SIZE, NUM_GENERATIONS, TOURNAMENT_SIZE,
#       CROSSOVER_RATE, MUTATION_RATE, MUTATION_SIGMA, ELITISM_COUNT,
#       NUM_EVAL_TRIALS
#   • Other ``config.py`` constants (e.g. GRAVITY) — best-effort only:
#       overrides land in the parent process but spawn-based workers
#       re-import config from disk, so deep physics constants won't take
#       effect inside the GA workers.
SWEEP: dict[str, Any] = {
    "variable": "NN_LAYERS",
    "values": [
        [6, 2],
        [6, 8, 2],
        [6, 8, 8, 2],
    ],
    "label": "Hidden layer size",
}

# Fixed validation / robustness seed sets — never regenerated, never patched
# onto ga_main.RANDOM_SEED. Passed explicitly into the evaluation function so
# every champion is scored on identical scenarios.
VALIDATION_SEEDS: list[int] = list(range(10_001, 10_051))    # 50 seeds
ROBUSTNESS_SEEDS: list[int] = list(range(20_001, 21_001))    # 1000 seeds
assert len(VALIDATION_SEEDS) == 50
assert len(ROBUSTNESS_SEEDS) == 1000
assert not set(VALIDATION_SEEDS) & set(ROBUSTNESS_SEEDS)


# ═══════════════════════════════════════════════════════════════════════
#  Output paths
# ═══════════════════════════════════════════════════════════════════════

RESULTS_DIR = "results"

# Filenames written into each per-run folder.
_DATA_FILENAME = "showcase_data.json"
_HEADLINE_FILENAME = "showcase_headline.json"

# Prefix used for auto-generated run-folder names (also used to discover
# previous runs when ``--plot-only`` is invoked without ``--run-dir``).
_RUN_DIR_PREFIX = "showcase_"

# Per-condition base names (suffix added in sweep mode).
_FITNESS_BASE = "showcase_fitness"
_LANDING_BASE = "showcase_landing"
_ROBUST_BASE = "showcase_robustness"

# Combined cross-condition base names.
_COMBINED_BEST_BASE = "showcase_combined_best_fitness"
_COMBINED_MEAN_BASE = "showcase_combined_mean_fitness"
_COMBINED_LAND_BASE = "showcase_combined_landing_rate"
_COMBINED_CDF_BASE = "showcase_combined_robustness_cdf"

# Window over which we compute the "final" landing/fitness numbers
# (averaging the tail trims single-generation noise without smearing
# across the learning curve).
FINAL_WINDOW = 10

# Robustness histogram domain.
ROBUST_FIT_MIN, ROBUST_FIT_MAX = 0.0, 180.0
ROBUST_BIN_WIDTH = 10.0
ROBUST_NUM_BINS = int(round((ROBUST_FIT_MAX - ROBUST_FIT_MIN) / ROBUST_BIN_WIDTH))


# Map of sweep-variable name → ``train()`` kwarg name. Variables in this map
# are passed through ``train()`` directly rather than being patched onto the
# config module (which spawn-based workers can't see).
_TRAIN_ARG_MAP: dict[str, str] = {
    "NN_LAYERS": "nn_layers",
    "POPULATION_SIZE": "population_size",
    "NUM_GENERATIONS": "num_generations",
    "TOURNAMENT_SIZE": "tournament_size",
    "CROSSOVER_RATE": "crossover_rate",
    "MUTATION_RATE": "mutation_rate",
    "MUTATION_SIGMA": "mutation_sigma",
    "ELITISM_COUNT": "elitism_count",
    "NUM_EVAL_TRIALS": "num_eval_trials",
}


# ═══════════════════════════════════════════════════════════════════════
#  Theme / styling
# ═══════════════════════════════════════════════════════════════════════
#
# Hand-picked accent palette: vivid enough for a dark slide, dark enough for a
# light slide. ``best`` / ``mean`` / ``land`` are matched per-theme; ``palette``
# is the sweep-overlay palette (cycled when there are more than six conditions).
_THEME = {
    "dark": {
        "fg":         "#f2f5ff",
        "muted":      "#a8b0cc",
        "grid":       "#a8b0cc",
        "best_color": "#00e5c8",
        "mean_color": "#ffd042",
        "land_color": "#7aeb7a",
        "fail_color": "#ff5c6f",
        "legend_bg":  "#1a1d2e",
        "palette": [
            "#00e5c8", "#ffd042", "#ff6b9d",
            "#7aeb7a", "#a07aff", "#ff9540",
        ],
    },
    "light": {
        "fg":         "#1a1d2e",
        "muted":      "#3a4055",
        "grid":       "#7a8094",
        "best_color": "#0d8c7c",
        "mean_color": "#c08410",
        "land_color": "#2a8b3c",
        "fail_color": "#a02030",
        "legend_bg":  "#ffffff",
        "palette": [
            "#0d8c7c", "#c08410", "#b03060",
            "#2a8b3c", "#604ab0", "#b85020",
        ],
    },
}


def _style_axes(ax: plt.Axes, theme: dict[str, str]) -> None:
    fg = theme["fg"]
    muted = theme["muted"]
    grid = theme["grid"]
    ax.tick_params(axis="both", colors=muted, labelsize=11, width=2, length=6)
    for spine in ax.spines.values():
        spine.set_color(muted)
        spine.set_linewidth(2)
    ax.grid(True, linestyle="--", linewidth=1.2, alpha=0.35, color=grid)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)


def _style_legend(leg, theme: dict[str, str]) -> None:
    leg.get_frame().set_facecolor(theme["legend_bg"])
    leg.get_frame().set_edgecolor(theme["fg"])
    leg.get_frame().set_linewidth(2)
    for text in leg.get_texts():
        text.set_color(theme["fg"])
        text.set_fontweight("bold")


def _save_transparent(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Filename helpers
# ═══════════════════════════════════════════════════════════════════════

def _value_label(value: Any) -> str:
    """Filename-safe label for a swept value.

    ``[16, 32, 16]`` → ``"16-32-16"``;  ``9.81`` → ``"9.81"``.
    """
    if isinstance(value, (list, tuple)):
        return "-".join(str(v) for v in value)
    return str(value)


def _condition_suffix(sweep_variable: str | None, value: Any) -> str:
    """Filename suffix for a single sweep condition (``""`` if no sweep)."""
    if sweep_variable is None:
        return ""
    var_short = str(sweep_variable).lower().replace("_", "")
    return f"_{var_short}_{_value_label(value)}"


def _per_condition_path(base: str, theme: str,
                        sweep_variable: str | None, value: Any,
                        run_dir: str) -> str:
    suffix = _condition_suffix(sweep_variable, value)
    return os.path.join(run_dir, theme, f"{base}_{theme}{suffix}.png")


def _combined_path(base: str, theme: str, run_dir: str) -> str:
    return os.path.join(run_dir, theme, f"{base}_{theme}.png")


# ═══════════════════════════════════════════════════════════════════════
#  Run-folder helpers
# ═══════════════════════════════════════════════════════════════════════

def _run_dir_descriptor() -> str:
    """Short tag for the current SWEEP setting, used in folder names."""
    var = SWEEP.get("variable")
    return "baseline" if var is None else f"sweep_{var}"


def make_run_dir_path() -> str:
    """Path of a fresh, never-overwriting run folder for the current SWEEP.

    Layout: ``results/showcase_<descriptor>_<YYYY-MM-DD_HH-MM-SS>``.
    """
    dt = time.strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(RESULTS_DIR,
                        f"{_RUN_DIR_PREFIX}{_run_dir_descriptor()}_{dt}")


def find_latest_run_dir() -> str | None:
    """Most recently modified ``results/showcase_*`` folder, or ``None``."""
    if not os.path.isdir(RESULTS_DIR):
        return None
    candidates = []
    for name in os.listdir(RESULTS_DIR):
        full = os.path.join(RESULTS_DIR, name)
        if name.startswith(_RUN_DIR_PREFIX) and os.path.isdir(full):
            candidates.append(full)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def data_json_path(run_dir: str) -> str:
    return os.path.join(run_dir, _DATA_FILENAME)


def headline_json_path(run_dir: str) -> str:
    return os.path.join(run_dir, _HEADLINE_FILENAME)


# ═══════════════════════════════════════════════════════════════════════
#  Champion-evaluation worker (top-level so it pickles for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════

def _eval_curriculum() -> dict:
    """Hardest curriculum entry — the deployment scenario for robustness."""
    last = config.CURRICULUM[-1]
    return {"gravity_var": last[1], "thrust_var": last[2], "max_wind": last[3]}


def _eval_worker(args):
    """Run one genome on a chunk of seeds. Returns (fits, succs).

    Each seed deterministically produces one random initial condition + one
    randomised physics realisation, using the *same sampling logic the GA
    uses for NUM_EVAL_TRIALS* (``make_sim`` + ``random_initial`` from
    ``main``). Curriculum is fixed at the hardest entry so robustness
    measurements are comparable across conditions.

    Success is defined exactly as the GA does it: ``sim.landed`` after the
    episode terminates (the physics module's LANDED_* thresholds).
    """
    genome, seeds, nn_layers, cur = args
    nn = NeuralNetwork(list(nn_layers))
    nn.set_genome(np.asarray(genome, dtype=float))
    fits: list[float] = []
    succs: list[bool] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        sim = ga_main.make_sim(cur, rng)
        initial = ga_main.random_initial(rng)
        sim.reset(initial)
        while True:
            inp = ga_main.normalize(sim.state)
            throttle, gimbal = nn.forward(inp)
            done, _info = sim.step(throttle, gimbal)
            if done:
                break
        fits.append(float(ga_main.compute_fitness(sim)))
        succs.append(bool(sim.landed))
    return fits, succs


def evaluate_genome_on_seeds(
    genome: np.ndarray,
    seeds: list[int],
    nn_layers: list[int],
    pool: ProcessPoolExecutor | None = None,
) -> tuple[list[float], list[bool]]:
    """Score a single genome on a fixed list of seeds (in parallel).

    If ``pool`` is provided, batches are dispatched onto it; otherwise a
    short-lived pool is created. Returns aligned ``(fitness_list,
    success_list)`` of length ``len(seeds)``.
    """
    seeds_list = list(seeds)
    n = len(seeds_list)
    if n == 0:
        return [], []

    n_workers = os.cpu_count() or 4
    chunk = max(1, n // n_workers)
    cur = _eval_curriculum()
    batches = [(genome, seeds_list[i:i + chunk], list(nn_layers), cur)
               for i in range(0, n, chunk)]

    def _consume(p: ProcessPoolExecutor):
        all_f: list[float] = []
        all_s: list[bool] = []
        for f, s in p.map(_eval_worker, batches):
            all_f.extend(f)
            all_s.extend(s)
        return all_f, all_s

    if pool is not None:
        return _consume(pool)
    with ProcessPoolExecutor(max_workers=n_workers) as new_pool:
        return _consume(new_pool)


# ═══════════════════════════════════════════════════════════════════════
#  Per-replicate champion selection + robustness measurement
# ═══════════════════════════════════════════════════════════════════════

def select_and_measure_champion(
    replicate_seed: int,
    top_genomes_last_10: list[dict],
    nn_layers: list[int],
    pool: ProcessPoolExecutor,
) -> dict:
    """Pick a champion from the last-10 best-of-generation genomes and
    stress-test it on the 1000-scenario robustness set.

    Returns a serialisable dict — see the ``champions`` schema in the
    module docstring at the top of this file.
    """
    if not top_genomes_last_10:
        raise ValueError("No candidate genomes provided for champion selection.")

    print(f"  champion: validating {len(top_genomes_last_10)} candidates "
          f"on {len(VALIDATION_SEEDS)} fixed seeds…")
    candidate_scores = []
    for cand in top_genomes_last_10:
        fits, _ = evaluate_genome_on_seeds(
            cand["genome"], VALIDATION_SEEDS, nn_layers, pool)
        candidate_scores.append(float(np.mean(fits)) if fits else float("-inf"))

    best_idx = int(np.argmax(candidate_scores))
    champion = top_genomes_last_10[best_idx]
    val_fitness = candidate_scores[best_idx]
    src_gen = int(champion["generation"])
    print(f"    champion from gen {src_gen}, "
          f"validation fitness = {val_fitness:.2f}")

    print(f"  robustness: scoring champion on "
          f"{len(ROBUSTNESS_SEEDS)} fixed seeds…")
    rob_fits, rob_succs = evaluate_genome_on_seeds(
        champion["genome"], ROBUSTNESS_SEEDS, nn_layers, pool)
    rob_arr = np.asarray(rob_fits)
    succ_rate = float(np.mean(rob_succs)) if rob_succs else 0.0
    print(f"    robustness fitness = {rob_arr.mean():.2f} ± {rob_arr.std():.2f}, "
          f"success rate = {succ_rate:.1%}")

    return {
        "replicate_seed": int(replicate_seed),
        "source_generation": src_gen,
        "validation_fitness": val_fitness,
        "robustness_fitness": [float(x) for x in rob_fits],
        "robustness_success": [bool(x) for x in rob_succs],
    }


# ═══════════════════════════════════════════════════════════════════════
#  Training (per replicate)
# ═══════════════════════════════════════════════════════════════════════

def _resolve_train_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    """Build the kwargs for ``ga_main.train`` for one replicate.

    Reads current values from ``config`` so that any best-effort overrides
    on the config module are reflected — and lets ``extra`` take final
    precedence (sweep-driven overrides for variables in ``_TRAIN_ARG_MAP``).
    """
    kwargs: dict[str, Any] = {
        "headless": True,
        "population_size": int(config.POPULATION_SIZE),
        "tournament_size": int(config.TOURNAMENT_SIZE),
        "crossover_rate": float(config.CROSSOVER_RATE),
        "mutation_rate": float(config.MUTATION_RATE),
        "mutation_sigma": float(config.MUTATION_SIGMA),
        "elitism_count": int(config.ELITISM_COUNT),
        "num_eval_trials": int(config.NUM_EVAL_TRIALS),
        "nn_layers": list(config.NN_LAYERS),
    }
    kwargs.update(extra)
    return kwargs


def _run_one_trial(
    seed: int,
    num_generations: int,
    train_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run baseline GA once with the given RNG seed.

    Note: ``seed`` is passed through ``train()``'s parameter explicitly
    rather than being patched onto ``ga_main.RANDOM_SEED`` (the latter has
    no effect once ``train``'s default is bound at import time).
    """
    kwargs = _resolve_train_kwargs(train_overrides)
    kwargs["seed"] = int(seed)
    kwargs["num_generations"] = int(num_generations)
    kwargs["run_id"] = f"showcase_seed{seed}"

    print(f"\n── Trial seed={seed}  "
          f"(pop={kwargs['population_size']}, gens={num_generations}, "
          f"trials/genome={kwargs['num_eval_trials']}, "
          f"layers={kwargs['nn_layers']}) ──")

    t0 = time.time()
    result = ga_main.train(**kwargs)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  "
          f"(final best={result['summary']['final_best_fitness']:.1f}, "
          f"final landed={result['summary']['final_landing_rate']:.1%})")
    return {
        "seed": int(seed),
        "elapsed_seconds": round(elapsed, 1),
        "generations": result["generations"],
        "summary": result["summary"],
        # In-memory only; never serialised to JSON.
        "_top_genomes_last_10": result.get("top_genomes_last_10", []),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Per-condition pipeline
# ═══════════════════════════════════════════════════════════════════════

def _override_for_condition(value: Any) -> tuple[dict[str, Any], list[int]]:
    """Apply a sweep override (if any) and return the train() extras + the
    effective NN layer list. Best-effort writes to ``config.<VAR>`` for
    variables outside ``_TRAIN_ARG_MAP``.
    """
    var = SWEEP.get("variable")
    extras: dict[str, Any] = {}
    if var is None:
        return extras, list(config.NN_LAYERS)

    if var in _TRAIN_ARG_MAP:
        extras[_TRAIN_ARG_MAP[var]] = value
    else:
        if hasattr(config, var):
            setattr(config, var, value)
            print(f"  (best-effort: set config.{var} = {value!r}; "
                  f"spawn-based GA workers may not see this override)")
        else:
            raise ValueError(
                f"SWEEP variable {var!r} is not a known config constant.")

    effective_layers = (list(value) if var == "NN_LAYERS"
                        else list(config.NN_LAYERS))
    return extras, effective_layers


def run_one_condition(
    value: Any,
    num_trials: int,
    num_generations: int,
) -> dict[str, Any]:
    """Train ``num_trials`` replicates, then select + stress-test a champion
    from each. Returns a single condition dict suitable for ``conditions[]``.
    """
    train_overrides, effective_layers = _override_for_condition(value)

    var = SWEEP.get("variable")
    if var is not None:
        print(f"\n══ CONDITION  {var} = {value!r}  ══")

    trials: list[dict[str, Any]] = []
    for s in range(1, num_trials + 1):
        trials.append(_run_one_trial(s, num_generations, train_overrides))

    print("\n  ── champion selection + robustness ──")
    n_workers = os.cpu_count() or 4
    champions: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as eval_pool:
        for trial in trials:
            print(f"  replicate seed={trial['seed']}")
            champ = select_and_measure_champion(
                replicate_seed=trial["seed"],
                top_genomes_last_10=trial["_top_genomes_last_10"],
                nn_layers=effective_layers,
                pool=eval_pool,
            )
            champions.append(champ)

    # Strip in-memory genome arrays from trial dicts before serialising.
    serialisable_trials = [
        {k: v for k, v in t.items() if not k.startswith("_")}
        for t in trials
    ]

    return {
        "value": value,
        "value_label": _value_label(value),
        "config": {
            "population_size": int(config.POPULATION_SIZE),
            "num_generations": int(num_generations),
            "tournament_size": int(config.TOURNAMENT_SIZE),
            "crossover_rate": float(config.CROSSOVER_RATE),
            "mutation_rate": float(config.MUTATION_RATE),
            "mutation_sigma": float(config.MUTATION_SIGMA),
            "elitism_count": int(config.ELITISM_COUNT),
            "num_eval_trials": int(config.NUM_EVAL_TRIALS),
            "nn_layers": list(effective_layers),
        },
        "num_trials": int(num_trials),
        "trials": serialisable_trials,
        "champions": champions,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Sweep orchestration
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(num_trials: int, num_generations: int) -> dict[str, Any]:
    """Run every condition in the sweep (or just one when sweep is off)."""
    var = SWEEP.get("variable")
    if var is None:
        # Single baseline condition. Use a single placeholder value so the
        # downstream code can treat everything as a list of conditions.
        conditions = [run_one_condition("baseline", num_trials, num_generations)]
        return {
            "sweep": None,
            "conditions": conditions,
        }

    values = list(SWEEP.get("values", []))
    if not values:
        raise ValueError("SWEEP['variable'] is set but SWEEP['values'] is empty.")

    # Snapshot config attributes that we may overwrite during best-effort
    # sweeps so we can restore them at the end.
    snapshot = {var: getattr(config, var)} if hasattr(config, var) else {}
    conditions: list[dict[str, Any]] = []
    try:
        for v in values:
            conditions.append(run_one_condition(v, num_trials, num_generations))
    finally:
        for k, original in snapshot.items():
            setattr(config, k, original)

    return {
        "sweep": {
            "variable": str(var),
            "values": values,
            "label": str(SWEEP.get("label", var)),
        },
        "conditions": conditions,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Save / load (with auto-detect for legacy single-condition format)
# ═══════════════════════════════════════════════════════════════════════

def save_data(data: dict[str, Any], run_dir: str) -> None:
    path = data_json_path(run_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  saved {path}")


def _wrap_legacy_data(legacy: dict[str, Any]) -> dict[str, Any]:
    """Promote the pre-sweep showcase_data.json schema to the current
    list-of-conditions schema. The single legacy condition has no champion
    data — it'll show up as an empty ``champions`` list, which the
    plotting/headline code tolerates.
    """
    return {
        "sweep": None,
        "conditions": [
            {
                "value": "baseline",
                "value_label": "baseline",
                "config": legacy.get("config", {}),
                "num_trials": legacy.get("num_trials",
                                         len(legacy.get("trials", []))),
                "trials": legacy.get("trials", []),
                "champions": legacy.get("champions", []),
            }
        ],
    }


def load_data(run_dir: str) -> dict[str, Any]:
    path = data_json_path(run_dir)
    with open(path) as f:
        raw = json.load(f)
    if "conditions" in raw:
        return raw
    if "trials" in raw:
        return _wrap_legacy_data(raw)
    raise ValueError(
        f"Unrecognised showcase data format in {path}: "
        "expected top-level 'conditions' or legacy 'trials' key.")


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation (per condition)
# ═══════════════════════════════════════════════════════════════════════

def _stack_metric(condition: dict[str, Any], metric: str) -> np.ndarray:
    arrs = []
    for t in condition["trials"]:
        arrs.append([row[metric] for row in t["generations"]])
    return np.asarray(arrs, dtype=float)


def aggregate(condition: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-generation mean / std across trials for each metric."""
    out: dict[str, np.ndarray] = {}
    for m in ("best_fitness", "mean_fitness", "landing_rate"):
        stack = _stack_metric(condition, m)
        out[f"{m}_per_trial"] = stack
        out[f"{m}_mean"] = stack.mean(axis=0)
        out[f"{m}_std"] = stack.std(axis=0)
    out["generation"] = np.asarray(
        [row["generation"] for row in condition["trials"][0]["generations"]],
        dtype=int,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Headline numbers (per condition)
# ═══════════════════════════════════════════════════════════════════════

def compute_headline(condition: dict[str, Any]) -> dict[str, Any]:
    cfg = condition["config"]
    agg = aggregate(condition)

    nn_params = NeuralNetwork.genome_size(cfg["nn_layers"])

    landing = agg["landing_rate_per_trial"]
    final_landing_per_trial = landing[:, -FINAL_WINDOW:].mean(axis=1)
    final_landing_mean = float(final_landing_per_trial.mean())
    final_landing_std = float(final_landing_per_trial.std())

    best = agg["best_fitness_per_trial"]
    final_best_per_trial = best[:, -FINAL_WINDOW:].mean(axis=1)
    initial_best_per_trial = best[:, 0]
    safe_initial = np.where(initial_best_per_trial > 0,
                            initial_best_per_trial, np.nan)
    improvement_per_trial = final_best_per_trial / safe_initial
    improvement_mean = float(np.nanmean(improvement_per_trial))
    improvement_std = float(np.nanstd(improvement_per_trial))

    final_best_mean = float(final_best_per_trial.mean())
    final_best_std = float(final_best_per_trial.std())
    initial_best_mean = float(initial_best_per_trial.mean())
    initial_best_std = float(initial_best_per_trial.std())

    landings_per_run = (cfg["population_size"]
                        * cfg["num_generations"]
                        * cfg["num_eval_trials"])

    h: dict[str, Any] = {
        "value": condition.get("value"),
        "value_label": condition.get("value_label", "baseline"),
        "num_trials": condition["num_trials"],
        "final_window_generations": FINAL_WINDOW,
        "nn_parameters_evolved": int(nn_params),
        "final_landing_rate_mean": final_landing_mean,
        "final_landing_rate_std": final_landing_std,
        "final_best_fitness_mean": final_best_mean,
        "final_best_fitness_std": final_best_std,
        "initial_best_fitness_mean": initial_best_mean,
        "initial_best_fitness_std": initial_best_std,
        "improvement_factor_mean": improvement_mean,
        "improvement_factor_std": improvement_std,
        "rocket_landings_per_training_run": int(landings_per_run),
    }

    champions = condition.get("champions", [])
    if champions:
        rep_means = np.array([np.mean(c["robustness_fitness"]) for c in champions])
        all_fits = np.concatenate(
            [np.asarray(c["robustness_fitness"]) for c in champions])
        rep_succ = np.array([np.mean(c["robustness_success"]) for c in champions])
        h.update({
            "champion_source_generations":
                [int(c["source_generation"]) for c in champions],
            "champion_validation_fitness":
                [float(c["validation_fitness"]) for c in champions],
            "robustness_fitness_mean": float(rep_means.mean()),
            "robustness_fitness_std": float(rep_means.std()),
            "robustness_fitness_median": float(np.median(all_fits)),
            "robustness_success_rate_mean": float(rep_succ.mean()),
            "robustness_success_rate_std": float(rep_succ.std()),
        })
    else:
        h.update({
            "champion_source_generations": [],
            "champion_validation_fitness": [],
            "robustness_fitness_mean": None,
            "robustness_fitness_std": None,
            "robustness_fitness_median": None,
            "robustness_success_rate_mean": None,
            "robustness_success_rate_std": None,
        })

    return h


def compute_all_headlines(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [compute_headline(c) for c in data["conditions"]]


def save_headlines(headlines: list[dict[str, Any]], run_dir: str) -> None:
    path = headline_json_path(run_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(headlines, f, indent=2)
    print(f"  saved {path}")


def print_headline_table(headlines: list[dict[str, Any]],
                         data: dict[str, Any]) -> None:
    sweep = data.get("sweep")
    label_col = sweep["label"] if sweep else "Condition"

    print()
    if len(headlines) == 1:
        h = headlines[0]
        win = h["final_window_generations"]
        nt = h["num_trials"]
        sep = "─" * 64
        print(sep)
        print(f"  HEADLINE NUMBERS  (mean ± std across {nt} independent seeds;")
        print(f"  'final' = average of the last {win} generations)")
        print(sep)
        print(f"  Neural-network parameters evolved : "
              f"{h['nn_parameters_evolved']:>10d}")
        print(f"  Final landing success rate        : "
              f"{h['final_landing_rate_mean']*100:>6.1f}% ± "
              f"{h['final_landing_rate_std']*100:.1f}%")
        print(f"  Final best fitness                : "
              f"{h['final_best_fitness_mean']:>6.1f}  ± "
              f"{h['final_best_fitness_std']:.1f}   "
              f"(gen 1: {h['initial_best_fitness_mean']:.1f} ± "
              f"{h['initial_best_fitness_std']:.1f})")
        print(f"  Improvement factor (final / gen 1): "
              f"{h['improvement_factor_mean']:>6.2f}× ± "
              f"{h['improvement_factor_std']:.2f}×")
        print(f"  Rocket landings simulated per run : "
              f"{h['rocket_landings_per_training_run']:>10,d}")
        if h.get("robustness_fitness_mean") is not None:
            print(f"  Champion source generations       : "
                  f"{h['champion_source_generations']}")
            print(f"  Robustness fitness (mean ± std)   : "
                  f"{h['robustness_fitness_mean']:>6.2f}  ± "
                  f"{h['robustness_fitness_std']:.2f}   "
                  f"(median {h['robustness_fitness_median']:.2f})")
            print(f"  Robustness success rate           : "
                  f"{h['robustness_success_rate_mean']*100:>6.1f}% ± "
                  f"{h['robustness_success_rate_std']*100:.1f}%")
        print(sep)
        return

    # Multi-condition summary table.
    print(f"  HEADLINE NUMBERS — {len(headlines)} conditions "
          f"({label_col})")
    header = (f"  {label_col:<24s} │ "
              f"{'Land%':>10s} │ {'Best fit':>11s} │ "
              f"{'Robust fit':>14s} │ {'Robust succ':>11s} │ "
              f"{'Champ gens':>16s}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for h in headlines:
        label = str(h.get("value_label") or h.get("value"))
        land = f"{h['final_landing_rate_mean']*100:5.1f}±{h['final_landing_rate_std']*100:.1f}%"
        best = f"{h['final_best_fitness_mean']:5.1f}±{h['final_best_fitness_std']:.1f}"
        if h.get("robustness_fitness_mean") is not None:
            rfit = (f"{h['robustness_fitness_mean']:5.1f}"
                    f"±{h['robustness_fitness_std']:.1f}")
            rsucc = (f"{h['robustness_success_rate_mean']*100:4.1f}"
                     f"±{h['robustness_success_rate_std']*100:.1f}%")
            gens = ",".join(str(g) for g in h["champion_source_generations"])
        else:
            rfit, rsucc, gens = "—", "—", "—"
        print(f"  {label:<24s} │ "
              f"{land:>10s} │ {best:>11s} │ "
              f"{rfit:>14s} │ {rsucc:>11s} │ "
              f"{gens:>16s}")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  Per-condition plots
# ═══════════════════════════════════════════════════════════════════════

def plot_fitness(condition: dict[str, Any],
                 theme_name: str, out_path: str) -> None:
    theme = _THEME[theme_name]
    agg = aggregate(condition)
    gens = agg["generation"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    best_mean = agg["best_fitness_mean"]
    best_std = agg["best_fitness_std"]
    mean_mean = agg["mean_fitness_mean"]
    mean_std = agg["mean_fitness_std"]

    ax.fill_between(gens, mean_mean - mean_std, mean_mean + mean_std,
                    color=theme["mean_color"], alpha=0.18, linewidth=0)
    ax.plot(gens, mean_mean, color=theme["mean_color"],
            linewidth=1.8, solid_capstyle="round",
            label="Mean population fitness")

    ax.fill_between(gens, best_mean - best_std, best_mean + best_std,
                    color=theme["best_color"], alpha=0.20, linewidth=0)
    ax.plot(gens, best_mean, color=theme["best_color"],
            linewidth=2.2, solid_capstyle="round",
            label="Best individual fitness")

    nt = condition["num_trials"]
    ax.set_xlabel("Generation", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Fitness (mean ± std across {nt} seeds)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Evolution learns: fitness over generations",
                 fontsize=15, fontweight="bold", pad=14)

    _style_axes(ax, theme)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=4))
    ax.set_xlim(gens.min(), gens.max())

    leg = ax.legend(loc="lower right", fontsize=11, framealpha=0.92)
    _style_legend(leg, theme)

    _save_transparent(fig, out_path)


def plot_landing(condition: dict[str, Any],
                 theme_name: str, out_path: str) -> None:
    theme = _THEME[theme_name]
    agg = aggregate(condition)
    gens = agg["generation"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    rate_mean = agg["landing_rate_mean"] * 100
    rate_std = agg["landing_rate_std"] * 100

    ax.fill_between(gens, rate_mean - rate_std, rate_mean + rate_std,
                    color=theme["land_color"], alpha=0.22, linewidth=0)
    ax.plot(gens, rate_mean, color=theme["land_color"],
            linewidth=2.2, solid_capstyle="round",
            label="Landing success rate")

    nt = condition["num_trials"]
    ax.set_xlabel("Generation", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Rockets landed (mean ± std across {nt} seeds)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Rockets actually land: success rate over generations",
                 fontsize=15, fontweight="bold", pad=14)

    _style_axes(ax, theme)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=4))
    ax.set_xlim(gens.min(), gens.max())
    ax.set_ylim(-2, 102)

    leg = ax.legend(loc="lower right", fontsize=11, framealpha=0.92)
    _style_legend(leg, theme)

    _save_transparent(fig, out_path)


def plot_robustness_histogram(condition: dict[str, Any],
                              theme_name: str, out_path: str) -> None:
    """Stacked-bar histogram of champion fitness on the 1000-seed robustness
    set, pooled across all replicates' champions for this condition.

    Bin heights are normalised to a *percentage of all scored scenarios* so
    the y-axis stays bounded (0–100%) regardless of how many replicates
    were run. Each bar splits into successful (green) bottom and failed
    (red) top segments.
    """
    theme = _THEME[theme_name]
    champions = condition.get("champions", [])
    if not champions:
        print(f"  (skip {out_path}: no champion data)")
        return

    fits = np.concatenate(
        [np.asarray(c["robustness_fitness"]) for c in champions])
    succ = np.concatenate(
        [np.asarray(c["robustness_success"], dtype=bool) for c in champions])

    bins = np.linspace(ROBUST_FIT_MIN, ROBUST_FIT_MAX, ROBUST_NUM_BINS + 1)
    succ_counts, _ = np.histogram(fits[succ], bins=bins)
    fail_counts, _ = np.histogram(fits[~succ], bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    n_reps = len(champions)
    n_total = len(fits)
    n_per_rep = n_total // n_reps if n_reps else 0
    scale = 100.0 / n_total if n_total else 0.0
    succ_pct = succ_counts * scale
    fail_pct = fail_counts * scale

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.bar(centers, succ_pct, width=width * 0.92,
           color=theme["land_color"], alpha=0.92,
           edgecolor=theme["legend_bg"], linewidth=0.8,
           label="Successful landing")
    ax.bar(centers, fail_pct, width=width * 0.92, bottom=succ_pct,
           color=theme["fail_color"], alpha=0.92,
           edgecolor=theme["legend_bg"], linewidth=0.8,
           label="Failed scenario")

    ax.set_xlabel("Champion fitness on robustness scenario",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel(f"% of scenarios (pooled across {n_reps} replicates)",
                  fontsize=13, fontweight="bold")
    ax.set_title(
        f"Champion robustness — {n_reps} replicates × "
        f"{n_per_rep:,} scenarios = {n_total:,} total",
        fontsize=15, fontweight="bold", pad=14)

    _style_axes(ax, theme)
    ax.set_xlim(ROBUST_FIT_MIN, ROBUST_FIT_MAX)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    # Headroom so the tallest stacked bar never collides with the legend
    # (anchored top-left, away from the high-fitness pile-up).
    max_total = float((succ_pct + fail_pct).max(initial=0.0))
    if max_total > 0:
        ax.set_ylim(0, max_total * 1.18)

    leg = ax.legend(loc="upper left", fontsize=11, framealpha=0.92)
    _style_legend(leg, theme)

    _save_transparent(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  Combined cross-condition plots
# ═══════════════════════════════════════════════════════════════════════

def _plot_combined_metric(
    conditions: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    theme_name: str,
    out_path: str,
    as_percent: bool = False,
    linewidth: float = 2.4,
) -> None:
    theme = _THEME[theme_name]
    palette = theme["palette"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    x_min, x_max = None, None
    for i, cond in enumerate(conditions):
        agg = aggregate(cond)
        gens = agg["generation"]
        mean = agg[f"{metric}_mean"]
        std = agg[f"{metric}_std"]
        scale = 100.0 if as_percent else 1.0
        color = palette[i % len(palette)]

        ax.fill_between(gens, (mean - std) * scale, (mean + std) * scale,
                        color=color, alpha=0.16, linewidth=0)
        ax.plot(gens, mean * scale, color=color,
                linewidth=linewidth, solid_capstyle="round",
                label=str(cond.get("value_label") or cond.get("value")))

        x_min = gens.min() if x_min is None else min(x_min, gens.min())
        x_max = gens.max() if x_max is None else max(x_max, gens.max())

    ax.set_xlabel("Generation", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

    _style_axes(ax, theme)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=4))
    if x_min is not None:
        ax.set_xlim(x_min, x_max)
    if as_percent:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(-2, 102)

    leg = ax.legend(loc="lower right", fontsize=11, framealpha=0.92,
                    title=str(SWEEP.get("label") or "Condition"))
    leg.get_title().set_color(theme["fg"])
    leg.get_title().set_fontweight("bold")
    _style_legend(leg, theme)

    _save_transparent(fig, out_path)


def plot_combined_robustness_cdf(
    conditions: list[dict[str, Any]],
    theme_name: str,
    out_path: str,
) -> None:
    """ECDF of pooled champion robustness fitness, one line per condition."""
    theme = _THEME[theme_name]
    palette = theme["palette"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    plotted = 0
    for i, cond in enumerate(conditions):
        champions = cond.get("champions", [])
        if not champions:
            continue
        fits = np.sort(np.concatenate(
            [np.asarray(c["robustness_fitness"]) for c in champions]))
        if fits.size == 0:
            continue
        # ECDF: F(x) = P(X ≤ x). Step plot from (-inf, 0) to (max, 1).
        y = np.arange(1, fits.size + 1) / fits.size
        # Anchor the curve to the left edge so the step starts cleanly.
        x = np.concatenate(([ROBUST_FIT_MIN], fits))
        y = np.concatenate(([0.0], y))
        color = palette[i % len(palette)]
        ax.step(x, y, where="post", color=color, linewidth=2.6,
                label=str(cond.get("value_label") or cond.get("value")))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"  (skip {out_path}: no champion data)")
        return

    ax.set_xlabel("Champion fitness on robustness scenario",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative fraction of scenarios",
                  fontsize=13, fontweight="bold")
    ax.set_title("Champion robustness CDF — lower-left = more failures",
                 fontsize=15, fontweight="bold", pad=14)

    _style_axes(ax, theme)
    ax.set_xlim(ROBUST_FIT_MIN, ROBUST_FIT_MAX)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    leg = ax.legend(loc="lower right", fontsize=11, framealpha=0.92,
                    title=str(SWEEP.get("label") or "Condition"))
    leg.get_title().set_color(theme["fg"])
    leg.get_title().set_fontweight("bold")
    _style_legend(leg, theme)

    _save_transparent(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════
#  Top-level plotting orchestration
# ═══════════════════════════════════════════════════════════════════════

def plot_all(data: dict[str, Any], run_dir: str) -> None:
    print("\nPlotting…")
    for theme in ("dark", "light"):
        os.makedirs(os.path.join(run_dir, theme), exist_ok=True)

    sweep = data.get("sweep")
    sweep_var = sweep["variable"] if sweep else None
    conditions = data["conditions"]

    for cond in conditions:
        value = cond.get("value")
        for theme in ("dark", "light"):
            plot_fitness(cond, theme,
                         _per_condition_path(_FITNESS_BASE, theme,
                                             sweep_var, value, run_dir))
            plot_landing(cond, theme,
                         _per_condition_path(_LANDING_BASE, theme,
                                             sweep_var, value, run_dir))
            plot_robustness_histogram(cond, theme,
                                      _per_condition_path(_ROBUST_BASE, theme,
                                                          sweep_var, value,
                                                          run_dir))

    if sweep is not None and len(conditions) > 1:
        for theme in ("dark", "light"):
            _plot_combined_metric(
                conditions, "best_fitness",
                ylabel="Best individual fitness (mean ± σ across replicates)",
                title="Best fitness over generations — by condition",
                theme_name=theme,
                out_path=_combined_path(_COMBINED_BEST_BASE, theme, run_dir),
            )
            _plot_combined_metric(
                conditions, "mean_fitness",
                ylabel="Mean population fitness (mean ± σ across replicates)",
                title="Mean fitness over generations — by condition",
                theme_name=theme,
                out_path=_combined_path(_COMBINED_MEAN_BASE, theme, run_dir),
            )
            _plot_combined_metric(
                conditions, "landing_rate",
                ylabel="Rockets landed (mean ± σ across replicates)",
                title="Landing rate over generations — by condition",
                theme_name=theme,
                out_path=_combined_path(_COMBINED_LAND_BASE, theme, run_dir),
                as_percent=True,
                # Landing curves converge into a tangle in the second half of
                # training, so use a noticeably thinner stroke than the
                # fitness combos (which have plenty of vertical separation).
                linewidth=1.6,
            )
            plot_combined_robustness_cdf(
                conditions, theme,
                _combined_path(_COMBINED_CDF_BASE, theme, run_dir),
            )


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate showcase graphs and headline numbers.")
    ap.add_argument("--trials", type=int, default=None,
                    help="number of independent seeds to train per condition "
                         "(default: 5 in baseline mode, 3 in sweep mode)")
    ap.add_argument("--generations", type=int, default=config.NUM_GENERATIONS,
                    help=f"generations per trial "
                         f"(default: {config.NUM_GENERATIONS})")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip training; re-plot from a saved run folder. "
                         "Uses --run-dir if given, otherwise the most recent "
                         f"{_RUN_DIR_PREFIX}* folder under {RESULTS_DIR}/.")
    ap.add_argument("--run-dir", default=None,
                    help="output folder (training mode) or input folder "
                         "(--plot-only mode). Defaults to a fresh, "
                         "datetime-stamped folder under "
                         f"{RESULTS_DIR}/ in training mode.")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    sweep_active = SWEEP.get("variable") is not None
    if args.trials is None:
        args.trials = 3 if sweep_active else 5

    if args.plot_only:
        if args.run_dir:
            run_dir = args.run_dir
        else:
            run_dir = find_latest_run_dir()
            if run_dir is None:
                raise SystemExit(
                    f"--plot-only: no {_RUN_DIR_PREFIX}* folder found under "
                    f"{RESULTS_DIR}/ and no --run-dir provided.")
            print(f"  (auto-selected most recent run folder: {run_dir})")
        if not os.path.isdir(run_dir):
            raise SystemExit(f"--plot-only: {run_dir} is not a directory.")
        data_path = data_json_path(run_dir)
        if not os.path.exists(data_path):
            raise SystemExit(
                f"--plot-only: {data_path} does not exist; "
                "run without --plot-only to generate it first.")
        data = load_data(run_dir)
        n_cond = len(data["conditions"])
        n_gens = (len(data["conditions"][0]["trials"][0]["generations"])
                  if data["conditions"] and data["conditions"][0]["trials"]
                  else 0)
        sweep = data.get("sweep")
        sweep_str = (f"sweep over {sweep['variable']} = "
                     f"{[c.get('value_label') for c in data['conditions']]}"
                     if sweep else "single condition (baseline)")
        print(f"Loaded {data_path}  ({n_cond} condition(s), "
              f"{n_gens} generations · {sweep_str})")
    else:
        run_dir = args.run_dir or make_run_dir_path()
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run folder: {run_dir}")
        data = run_pipeline(num_trials=args.trials,
                            num_generations=args.generations)
        save_data(data, run_dir)

    headlines = compute_all_headlines(data)
    save_headlines(headlines, run_dir)
    print_headline_table(headlines, data)
    plot_all(data, run_dir)
    print()


if __name__ == "__main__":
    main()
