"""Pool the test results of all rotations of a build.

Example:
    uv run python -m scripts.real_data.summarize --dataset hematopoiesis
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.real_data import models as M
from scripts.real_data.run_analyses import WINDOWS


def mean_sd(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return (
        {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values)),
            "n_rotations": len(values),
        }
        if values
        else None
    )


def lineage_bootstrap(df, stat, n_boot=1000):
    """95% interval of ``stat`` when whole lineages (clones or families) are resampled."""
    rng = np.random.default_rng()
    rows_of = list(df.groupby("lineage").indices.values())
    values = []
    for _ in range(n_boot):
        pick = rng.integers(len(rows_of), size=len(rows_of))
        value = stat(df.iloc[np.concatenate([rows_of[k] for k in pick])])
        if value is not None and np.isfinite(value):
            values.append(value)
    return (
        [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        if values
        else None
    )


def _accuracy(d):
    return float(((d["prob"] >= 0.5).astype(int) == d["label"]).mean())


def _balanced_accuracy(d):
    from sklearn.metrics import balanced_accuracy_score

    return (
        float(balanced_accuracy_score(d["label"], (d["prob"] >= 0.5).astype(int)))
        if d["label"].nunique() == 2
        else None
    )


def _auc(d):
    from sklearn.metrics import roc_auc_score

    return (
        float(roc_auc_score(d["label"], d["prob"]))
        if d["label"].nunique() == 2
        else None
    )


def pooled(paths):
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    out = M.classification_metrics(df["label"].to_numpy(), df["prob"].to_numpy())
    out["lineages"] = int(df["lineage"].nunique())
    out["ci95"] = {
        "accuracy": lineage_bootstrap(df, _accuracy),
        "balanced_accuracy": lineage_bootstrap(df, _balanced_accuracy),
        "auc": lineage_bootstrap(df, _auc),
    }
    out["lineage_level"] = M.lineage_level(
        df["label"].to_numpy(), df["prob"].to_numpy(), df["lineage"].to_numpy()
    )
    return out


def check_single_build(rot_dirs, results):
    """Check that all rotations and their results come from the same build."""
    metas = [json.loads((d / "dataset.json").read_text()) for d in rot_dirs]
    built = {m.get("run_id") for m in metas}
    trained = {r.get("run_id") for r in results}
    if len(built) != 1:
        raise SystemExit(
            f"The rotation datasets come from different builds {sorted(map(str, built))}. "
            "Rebuild them with prepare_dataset.py, then rerun run_analyses.py for every rotation."
        )
    if trained != built:
        raise SystemExit(
            f"Some results come from an older build {sorted(map(str, trained - built))}. "
            "Rerun run_analyses.py for every rotation."
        )
    n_groups = metas[0]["n_groups"]
    if len(rot_dirs) < n_groups:
        print(
            f"Note: pooling {len(rot_dirs)} of {n_groups} rotations, so only part of the lineages are tested."
        )
    return built.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(WINDOWS), required=True)
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()
    base = Path(args.runs_dir) / args.dataset
    rot_dirs = sorted(p.parent for p in base.glob("rotation_*/results.json"))
    if not rot_dirs:
        raise SystemExit(f"No results found in {base}; run run_analyses.py first.")
    results = [json.loads((d / "results.json").read_text()) for d in rot_dirs]
    run_id = check_single_build(rot_dirs, results)
    summary = {"dataset": args.dataset, "run_id": run_id, "rotations": len(results)}

    summary["autoencoder_test_reconstruction_r"] = mean_sd(
        [r["autoencoder"]["test_reconstruction_pearson_r"] for r in results]
    )

    summary["classification"] = {}
    for name in WINDOWS[args.dataset]:
        tag = name.replace(" ", "_")
        summary["classification"][name] = {
            "pooled": pooled([d / f"pred_classification_{tag}.csv" for d in rot_dirs]),
            "accuracy_per_rotation": mean_sd(
                [r["classification"][name].get("accuracy") for r in results]
            ),
            "auc_per_rotation": mean_sd(
                [r["classification"][name].get("auc") for r in results]
            ),
        }

    summary["regression"] = {}
    tasks = dict.fromkeys(n for r in results for n in r.get("regression", {}))
    for name in tasks:
        entries = [
            r["regression"][name]
            for r in results
            if "series_r" in r.get("regression", {}).get(name, {})
        ]
        if not entries:
            continue
        r_all = np.concatenate([e["series_r"] for e in entries])
        r_rand = np.concatenate([e["series_r_random"] for e in entries])
        labels = np.concatenate([e["series_label"] for e in entries])
        classes = results[0]["classes"]
        per_class = {}
        for c, cls in enumerate(classes):
            m = labels == c
            if m.any():
                per_class[cls] = {
                    "n": int(m.sum()),
                    "pearson_r_mean": float(np.nanmean(r_all[m])),
                    "gene_mean_R2_per_rotation": mean_sd(
                        [
                            e["per_class"].get(cls, {}).get("gene_mean_R2")
                            for e in entries
                        ]
                    ),
                }
        frame = pd.DataFrame(
            {
                "lineage": np.concatenate([e["series_lineage"] for e in entries]),
                "r": r_all,
                "r_random": r_rand,
            }
        )
        summary["regression"][name] = {
            "n": len(r_all),
            "lineages": int(frame["lineage"].nunique()),
            "pearson_r_mean": float(np.nanmean(r_all)),
            "pearson_r_random_cells_mean": float(np.nanmean(r_rand)),
            "ci95": {
                "pearson_r_mean": lineage_bootstrap(
                    frame, lambda d: float(np.nanmean(d["r"]))
                ),
                "difference_vs_random_cells": lineage_bootstrap(
                    frame,
                    lambda d: float(np.nanmean(d["r"]) - np.nanmean(d["r_random"])),
                ),
            },
            "per_class": per_class,
        }

    summary["ablation"] = {
        "window": results[0]["ablation"]["window"],
        "pooled_on_clonal_test_series": pooled(
            [d / "pred_ablation.csv" for d in rot_dirs]
        ),
    }

    pert = [
        r["perturbation"]
        for r in results
        if "early_classifier" in r.get("perturbation", {})
    ]
    if pert:
        agg = {}
        for readout in ("early_classifier", "forward_simulation"):
            src = sum(p[readout]["predicted_source_control"] for p in pert)
            sw = sum(p[readout]["switched"] for p in pert)
            agg[readout] = {
                "series": sum(p[readout]["series"] for p in pert),
                "predicted_source_control": src,
                "switched": sw,
                "switched_fraction_of_predicted_source": sw / max(1, src),
            }
        sims = defaultdict(list)
        for p in pert:
            n = p["forward_simulation"]["switched"]
            for key in (
                "similarity_switched_vs_real_target",
                "similarity_switched_vs_real_source",
                "similarity_real_target_vs_real_source",
            ):
                if key in p:
                    sims[key].append((n, p[key]["pearson_r"], p[key]["r2_score"]))
        for key, rows in sims.items():
            w = np.array([n for n, _, _ in rows], dtype=float)
            agg[key] = {
                "pearson_r_weighted_mean": float(
                    np.average([r for _, r, _ in rows], weights=w if w.sum() else None)
                ),
                "r2_score_weighted_mean": float(
                    np.average([q for _, _, q in rows], weights=w if w.sum() else None)
                ),
                "rotations": len(rows),
            }
        gene_votes = defaultdict(lambda: [0, 0])
        for p in pert:
            for g in p.get("genes_overexpressed", []):
                gene_votes[g][0] += 1
            for g in p.get("genes_knocked_out", []):
                gene_votes[g][1] += 1
        agg["perturbed_genes_by_rotation_count"] = dict(
            sorted(
                {
                    g: {"overexpressed": v[0], "knocked_out": v[1]}
                    for g, v in gene_votes.items()
                }.items(),
                key=lambda kv: -sum(kv[1].values()),
            )
        )
        summary["perturbation"] = {
            "source": pert[0]["source"],
            "target": pert[0]["target"],
            "day": pert[0]["day"],
            **agg,
        }

    top = defaultdict(lambda: defaultdict(list))
    for r in results:
        for day, entry in r.get("top_genes_by_day", {}).items():
            scale = np.max(np.abs(entry["scores"])) or 1.0
            for g, s in zip(entry["genes"], entry["scores"]):
                top[day][g].append(s / scale)
    summary["top_genes_by_day"] = {
        day: [
            g
            for g, _ in sorted(scores.items(), key=lambda kv: -abs(np.sum(kv[1])))[:10]
        ]
        for day, scores in top.items()
    }
    combined = defaultdict(float)
    for scores in top.values():
        for g, s in scores.items():
            combined[g] += abs(np.sum(s))
    summary["top_200_genes"] = [
        g for g, _ in sorted(combined.items(), key=lambda kv: -kv[1])[:200]
    ]

    if args.dataset == "reprogramming":
        summary["barcoding"] = {
            label: pooled([d / f"pred_barcoding_{label}.csv" for d in rot_dirs])
            for label in ("cumulative", "single")
        }

    (base / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("top_200_genes",)},
            indent=1,
            default=str,
        )[:12000]
    )
    print(f"Saved {base / 'summary.json'}")


if __name__ == "__main__":
    main()
