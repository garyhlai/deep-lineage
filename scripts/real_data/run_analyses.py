"""Train and evaluate Deep Lineage on one rotation of a real dataset.

Example:
    uv run python -m scripts.real_data.run_analyses --dataset hematopoiesis --rotation 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.real_data import models as M
from scripts.real_data.data import MISSING, load_dataset

# Fate-prediction windows: number of leading days given to the classifier.
WINDOWS = {
    "hematopoiesis": {"Day 2": 1, "Days 2-4": 2, "All days": 3},
    "reprogramming": {"Days 6-12": 3, "Days 6-15": 4, "Days 6-21": 5, "All days": 6},
}
MAIN_MODE = {"hematopoiesis": "clonal", "reprogramming": "cumulative"}
# Expression-prediction tasks: (index of the predicted day, indices of the input days).
REGRESSION_TASKS = {
    "hematopoiesis": {
        "Day 4 from days 2 and 6": (1, [0, 2]),
        "Day 6 from days 2 and 4": (2, [0, 1]),
        "Day 4 from day 2": (1, [0]),
    },
    "reprogramming": {
        "Day 12": (2, [0, 1]),
        "Day 15": (3, [0, 1, 2]),
        "Day 21": (4, [0, 1, 2, 3]),
        "Day 28": (5, [0, 1, 2, 3, 4]),
    },
}
ABLATION_WINDOW = {"hematopoiesis": "Day 2", "reprogramming": "Days 6-21"}
# In silico perturbation: classifier window used for attribution, perturbed day,
# starting fate (source) and desired fate (target), and the forward regressors.
PERTURBATION = {
    "hematopoiesis": {
        "window": "Day 2",
        "day": 0,
        "source": "Neutrophil",
        "target": "Monocyte",
        "forward": ["Day 4 from day 2", "Day 6 from days 2 and 4"],
    },
    "reprogramming": {
        "window": "Days 6-15",
        "day": 3,
        "source": "Failed",
        "target": "Reprogrammed",
        "forward": ["Day 21", "Day 28"],
    },
}
MIN_VAL_SERIES = 50
MAX_ATTRIBUTION_SERIES = 2000


def last_observed_day(idx):
    """Index of the last observed day of each series (the day of its outcome cell)."""
    observed = idx != MISSING
    return observed.shape[1] - 1 - np.argmax(observed[:, ::-1], axis=1)


def window_arrays(s, Z, n_days):
    """Inputs for a fate-prediction window of the first ``n_days`` days. For early
    windows, only series whose outcome cell (the last observed cell) comes after the
    window are used; the all-days window keeps every series."""
    idx = s["idx"]
    keep = (idx[:, :n_days] != MISSING).any(axis=1)
    if n_days < idx.shape[1]:
        keep &= last_observed_day(idx) >= n_days
    return (
        M.series_latent(idx[keep][:, :n_days], Z),
        s["label"][keep],
        s["lineage"][keep],
        keep,
    )


def task_arrays(s, Z, X, target, inputs):
    """Inputs (latent embeddings of the input days only) and target expression for a
    series-to-expression task; series lacking the target day or all input days are skipped."""
    idx = s["idx"]
    keep = (idx[:, target] != MISSING) & (idx[:, inputs] != MISSING).any(axis=1)
    idx_in = np.full_like(idx[keep], MISSING)
    idx_in[:, inputs] = idx[keep][:, inputs]
    return (
        M.series_latent(idx_in, Z),
        X[idx[keep, target]],
        s["label"][keep],
        s["lineage"][keep],
    )


SERIES_KEYS = ("idx", "label", "lineage")


def train_and_validation(series):
    """Validation series for early stopping. If the validation fold is too small, part
    of the training lineages (never the test lineages) is held out instead."""
    tr, va = series["train"], series["val"]
    if len(va["label"]) >= MIN_VAL_SERIES and len(set(va["label"].tolist())) == 2:
        return tr, va
    merged = {k: np.concatenate([tr[k], va[k]]) for k in SERIES_KEYS}
    lineages = np.array(sorted(set(merged["lineage"].tolist())))
    rng = np.random.default_rng()
    held = set(
        rng.choice(lineages, size=max(1, round(0.15 * len(lineages))), replace=False)
    )
    is_val = np.array([lin in held for lin in merged["lineage"]])
    return (
        {k: v[~is_val] for k, v in merged.items()},
        {k: v[is_val] for k, v in merged.items()},
    )


def ensure_validation(Xtr, ytr, ltr, Xva, yva, rng, fraction=0.1):
    """Guard for windows in which no validation series has an observed day: hold out
    a random part of the training lineages instead."""
    if len(yva) > 0:
        return Xtr, ytr, Xva, yva
    lineages = np.unique(ltr)
    held = rng.choice(
        lineages, size=max(1, round(fraction * len(lineages))), replace=False
    )
    is_val = np.isin(ltr, held)
    return Xtr[~is_val], ytr[~is_val], Xtr[is_val], ytr[is_val]


def non_clonal_series(s, cells, classes, days, rng):
    """Ablation: keep each series' observed days but draw every cell at random from the
    cells of that day in the series' lineages, ignoring clones. The label comes from
    the newly drawn last cell."""
    idx = s["idx"].copy()
    in_pool = cells["lineage"].isin(set(s["lineage"].tolist())).to_numpy()
    labelled = cells["state"].isin(classes).to_numpy()
    pool = {
        d: np.flatnonzero(in_pool & (cells["day"] == day).to_numpy())
        for d, day in enumerate(days)
    }
    labels, lineages = [], []
    for i in range(len(idx)):
        present = np.flatnonzero(idx[i] != MISSING)
        for d in present:
            candidates = pool[d][labelled[pool[d]]] if d == present[-1] else pool[d]
            idx[i, d] = rng.choice(candidates)
        labels.append(classes.index(cells["state"].iat[idx[i, present[-1]]]))
        lineages.append(f"random_{i}")
    return {"idx": idx, "label": np.array(labels), "lineage": np.array(lineages)}


def save_results(run_dir, results, t0):
    results["seconds"] = round(time.time() - t0, 1)
    (run_dir / "results.json").write_text(json.dumps(results, indent=1))
    print(f"Saved {run_dir / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(WINDOWS), required=True)
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--shap_samples",
        type=int,
        default=32,
        help="Background samples per series for the SHAP values (default: 32)",
    )
    parser.add_argument(
        "--fate_only",
        action="store_true",
        help="Run only the autoencoder and the fate analyses "
        "(classifiers, ablation and barcoding comparison)",
    )
    args = parser.parse_args()
    t0 = time.time()
    run_dir = Path(args.runs_dir) / args.dataset / f"rotation_{args.rotation}"
    data = load_dataset(run_dir / "dataset.npz")
    for old in [*run_dir.glob("pred_*.csv"), run_dir / "results.json"]:
        old.unlink(missing_ok=True)  # outputs of an earlier run of this rotation
    X, cells, meta = data["X"], data["cells"], data["meta"]
    classes, days, genes = meta["classes"], meta["days"], np.array(meta["genes"])
    series = data["series"][MAIN_MODE[args.dataset]]
    rng = np.random.default_rng()
    results = {
        "run_id": meta["run_id"],
        "dataset": args.dataset,
        "rotation": args.rotation,
        "classes": classes,
    }

    # ---------------------------------------------------------------- autoencoder
    fold = cells["fold"].to_numpy()
    ae = M.train_autoencoder(
        X[fold == "train"],
        X[fold == "val"],
        epochs=args.epochs,
        patience=args.patience,
    )
    Z = M.encode_genes(ae.encoder, X)
    recon = ae.model.predict(X[fold == "test"], batch_size=4096, verbose=0)
    results["autoencoder"] = {
        "train_cells": int((fold == "train").sum()),
        "test_cells": int((fold == "test").sum()),
        "test_reconstruction_pearson_r": float(
            np.nanmean(M.rowwise_pearson(recon, X[fold == "test"]))
        ),
    }
    print(f"[{time.time() - t0:.0f}s] autoencoder done", flush=True)

    # ---------------------------------------------------------------- fate prediction
    tr, va = train_and_validation(series)
    te = series["test"]
    classifiers, results["classification"] = {}, {}
    for name, n_days in WINDOWS[args.dataset].items():
        Xtr, ytr, ltr, _ = window_arrays(tr, Z, n_days)
        Xva, yva, _, _ = window_arrays(va, Z, n_days)
        Xte, yte, lte, _ = window_arrays(te, Z, n_days)
        Xtr, ytr, Xva, yva = ensure_validation(Xtr, ytr, ltr, Xva, yva, rng)
        clf = M.build_classifier(n_days, Z.shape[1])
        epochs = M.fit(
            clf,
            Xtr,
            M.one_hot(ytr),
            Xva,
            M.one_hot(yva),
            args.epochs,
            args.patience,
            class_weight=M.balanced_class_weight(ytr),
        )
        classifiers[name] = clf
        entry = {"train_series": len(ytr), "epochs": epochs}
        if len(yte):
            prob = clf.predict(Xte, verbose=0)[:, 1]
            entry.update(M.classification_metrics(yte, prob))
            entry["lineage_level"] = M.lineage_level(yte, prob, lte)
            pd.DataFrame({"lineage": lte, "label": yte, "prob": prob}).to_csv(
                run_dir / f"pred_classification_{name.replace(' ', '_')}.csv",
                index=False,
            )
        results["classification"][name] = entry
    print(f"[{time.time() - t0:.0f}s] classifiers done", flush=True)

    # ---------------------------------------------------------------- ablation
    window = ABLATION_WINDOW[args.dataset]
    n_days = WINDOWS[args.dataset][window]
    rtr = non_clonal_series(tr, cells, classes, days, rng)
    rva = non_clonal_series(va, cells, classes, days, rng)
    Xtr, ytr, ltr, _ = window_arrays(rtr, Z, n_days)
    Xva, yva, _, _ = window_arrays(rva, Z, n_days)
    Xtr, ytr, Xva, yva = ensure_validation(Xtr, ytr, ltr, Xva, yva, rng)
    abl = M.build_classifier(n_days, Z.shape[1])
    M.fit(
        abl,
        Xtr,
        M.one_hot(ytr),
        Xva,
        M.one_hot(yva),
        args.epochs,
        args.patience,
        class_weight=M.balanced_class_weight(ytr),
    )
    Xte, yte, lte, _ = window_arrays(te, Z, n_days)
    results["ablation"] = {"window": window}
    if len(yte):
        prob = abl.predict(Xte, verbose=0)[:, 1]
        results["ablation"]["clonal_test_series"] = M.classification_metrics(yte, prob)
        pd.DataFrame({"lineage": lte, "label": yte, "prob": prob}).to_csv(
            run_dir / "pred_ablation.csv", index=False
        )
    print(f"[{time.time() - t0:.0f}s] ablation done", flush=True)

    # ---------------------------------------------------------------- barcoding resolution (reprogramming)
    if args.dataset == "reprogramming":
        name = "Days 6-15"
        n_days = WINDOWS[args.dataset][name]
        str_, sva = train_and_validation(data["series"]["single"])
        Xtr, ytr, ltr, _ = window_arrays(str_, Z, n_days)
        Xva, yva, _, _ = window_arrays(sva, Z, n_days)
        Xtr, ytr, Xva, yva = ensure_validation(Xtr, ytr, ltr, Xva, yva, rng)
        clf_single = M.build_classifier(n_days, Z.shape[1])
        M.fit(
            clf_single,
            Xtr,
            M.one_hot(ytr),
            Xva,
            M.one_hot(yva),
            args.epochs,
            args.patience,
            class_weight=M.balanced_class_weight(ytr),
        )
        Xte, yte, lte, _ = window_arrays(te, Z, n_days)
        results["barcoding"] = {
            "window": name,
            "test": "cumulative-barcoding test series",
        }
        if len(yte):
            for label, model in [
                ("cumulative", classifiers[name]),
                ("single", clf_single),
            ]:
                prob = model.predict(Xte, verbose=0)[:, 1]
                results["barcoding"][label] = M.classification_metrics(yte, prob)
                pd.DataFrame({"lineage": lte, "label": yte, "prob": prob}).to_csv(
                    run_dir / f"pred_barcoding_{label}.csv", index=False
                )
        print(f"[{time.time() - t0:.0f}s] barcoding comparison done", flush=True)

    if args.fate_only:
        save_results(run_dir, results, t0)
        return

    # ---------------------------------------------------------------- expression prediction
    regressors, results["regression"] = {}, {}
    for name, (target, inputs) in REGRESSION_TASKS[args.dataset].items():
        Xtr, ytr, _, _ = task_arrays(tr, Z, X, target, inputs)
        Xva, yva, _, _ = task_arrays(va, Z, X, target, inputs)
        Xte, yte, lab_te, lin_te = task_arrays(te, Z, X, target, inputs)
        if len(ytr) < 20 or len(yva) < 5:
            results["regression"][name] = {"skipped": "too few series"}
            continue
        reg = M.build_regressor(len(days), Z.shape[1], X.shape[1])
        epochs = M.fit(reg, Xtr, ytr, Xva, yva, args.epochs, args.patience)
        regressors[name] = reg
        entry = {"train_series": len(ytr), "epochs": epochs}
        if len(yte):
            pred = reg.predict(Xte, verbose=0)
            reference = X[(fold == "test") & (cells["day"].to_numpy() == days[target])]
            entry.update(
                M.regression_metrics(
                    yte, pred, lab_te, classes, reference, rng, lineages=lin_te
                )
            )
        results["regression"][name] = entry
    print(f"[{time.time() - t0:.0f}s] regressors done", flush=True)

    # ---------------------------------------------------------------- attribution and in silico perturbation
    p = PERTURBATION[args.dataset]
    src, tgt = classes.index(p["source"]), classes.index(p["target"])
    day_cells = lambda d, f: np.flatnonzero(
        (fold == f) & (cells["day"].to_numpy() == days[d])
    )
    results["perturbation"] = {
        "source": p["source"],
        "target": p["target"],
        "day": days[p["day"]],
    }

    def training_series(mask):
        """Training series used for attribution (at most MAX_ATTRIBUTION_SERIES)."""
        rows = np.flatnonzero(mask)
        if len(rows) > MAX_ATTRIBUTION_SERIES:
            rows = np.sort(rng.choice(rows, size=MAX_ATTRIBUTION_SERIES, replace=False))
        return tr["idx"][rows]

    if all(f in regressors for f in p["forward"]):
        n_w = WINDOWS[args.dataset][p["window"]]
        gmodel = M.gene_space_classifier(
            ae.encoder, classifiers[p["window"]], n_w, X.shape[1]
        )

        # Genes are ranked on training series; the perturbation is applied to test
        # series of the starting fate whose outcome cell comes after the window.
        def eligible(s):
            return (
                (s["label"] == src)
                & (s["idx"][:, p["day"]] != MISSING)
                & (last_observed_day(s["idx"]) >= n_w)
            )

        attr_idx = training_series(eligible(tr))
        keep = eligible(te)
        idx = te["idx"][keep]
        if len(attr_idx) and len(idx):
            attr = M.day_shap_values(
                gmodel,
                M.series_genes(attr_idx, X)[:, :n_w],
                p["day"],
                X[day_cells(p["day"], "train")],
                tgt,
                n_samples=args.shap_samples,
            )
            score = attr.mean(axis=0)
            top = np.argsort(-np.abs(score))[:10]
            up, down = top[score[top] > 0], top[score[top] < 0]
            results["perturbation"]["genes_overexpressed"] = genes[up].tolist()
            results["perturbation"]["genes_knocked_out"] = genes[down].tolist()
            G = M.series_genes(idx, X)

            def simulate(G_series):
                Zs = M.encode_series_genes(ae.encoder, G_series)
                Zs[:, p["day"] + 1 :] = 0.0
                for name in p["forward"]:
                    target_day, _ = REGRESSION_TASKS[args.dataset][name]
                    Zin = Zs.copy()
                    Zin[:, target_day:] = 0.0
                    pred = regressors[name].predict(Zin, verbose=0)
                    Zs[:, target_day] = M.encode_genes(ae.encoder, pred)
                    last_pred = pred
                prob = classifiers["All days"].predict(Zs, verbose=0)[:, tgt]
                return prob, last_pred

            def shift(p_ctrl, p_pert):
                source_ctrl = p_ctrl < 0.5
                switched = source_ctrl & (p_pert >= 0.5)
                return {
                    "series": len(p_ctrl),
                    "predicted_source_control": int(source_ctrl.sum()),
                    "switched": int(switched.sum()),
                    "switched_fraction_of_predicted_source": float(
                        switched.sum() / max(1, source_ctrl.sum())
                    ),
                    "target_fraction_control": float((p_ctrl >= 0.5).mean()),
                    "target_fraction_perturbed": float((p_pert >= 0.5).mean()),
                    "mean_target_probability_change": float((p_pert - p_ctrl).mean()),
                }, switched

            Gp = G.copy()
            Gp[:, p["day"]] = M.perturb(G[:, p["day"]], up, down)
            # Direct readout: early fate-bias classifier on the perturbed day.
            early_ctrl = gmodel.predict(G[:, :n_w], verbose=0)[:, tgt]
            early_pert = gmodel.predict(Gp[:, :n_w], verbose=0)[:, tgt]
            # Paper procedure: regress the later days from the perturbed series, then classify.
            p_ctrl, _ = simulate(G)
            p_pert, expr_pert = simulate(Gp)
            entry = {"early_classifier": shift(early_ctrl, early_pert)[0]}
            entry["forward_simulation"], switched = shift(p_ctrl, p_pert)
            last_day = REGRESSION_TASKS[args.dataset][p["forward"][-1]][0]
            real = lambda state: X[
                (cells["day"].to_numpy() == days[last_day])
                & (cells["state"].to_numpy() == state)
            ]
            real_t, real_s = real(p["target"]), real(p["source"])
            if switched.any() and len(real_t) and len(real_s):
                mean_switched = expr_pert[switched].mean(axis=0)
                entry["similarity_switched_vs_real_target"] = M.profile_similarity(
                    mean_switched, real_t.mean(axis=0)
                )
                entry["similarity_switched_vs_real_source"] = M.profile_similarity(
                    mean_switched, real_s.mean(axis=0)
                )
                entry["similarity_real_target_vs_real_source"] = M.profile_similarity(
                    real_s.mean(axis=0), real_t.mean(axis=0)
                )
            results["perturbation"].update(entry)

    # Top genes per day by mean SHAP value for the full-sequence classifier (toward
    # the target fate), computed on training series.
    gfull = M.gene_space_classifier(
        ae.encoder, classifiers["All days"], len(days), X.shape[1]
    )
    results["top_genes_by_day"] = {}
    for d in range(len(days)):
        attr_idx = training_series(tr["idx"][:, d] != MISSING)
        background = X[day_cells(d, "train")]
        if len(attr_idx) < 5 or len(background) < 5:
            continue
        G = M.series_genes(attr_idx, X)
        attr = M.day_shap_values(
            gfull, G, d, background, tgt, n_samples=args.shap_samples
        )
        score = attr.mean(axis=0)
        order = np.argsort(-np.abs(score))[:200]
        results["top_genes_by_day"][days[d]] = {
            "genes": genes[order].tolist(),
            "scores": score[order].tolist(),
        }
    print(f"[{time.time() - t0:.0f}s] attribution and perturbation done", flush=True)
    save_results(run_dir, results, t0)


if __name__ == "__main__":
    main()
