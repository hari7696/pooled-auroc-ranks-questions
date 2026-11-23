"""Stage 5: decomposition, downstream metrics and summary tables across all runs.

Writes CSVs to outputs/results/.  Usage: python src/analyze.py [--suffix _L2_judge] [--boot 300]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import OUT
from concurrent.futures import ProcessPoolExecutor

from metrics import aurc, best_of_n, bootstrap_decompose, decompose

DETECTORS = {
    # key: (display name, family, question-level by construction?)
    "semantic_entropy": ("Semantic entropy", "consistency", True),
    "num_clusters": ("No. of clusters", "consistency", True),
    "probe_question": ("Question probe", "probe", True),
    "question_text_probe": ("Question-text probe", "baseline", True),
    "neg_cluster_freq": ("Cluster frequency", "consistency", False),
    "neg_lexical_sim": ("Lexical similarity", "consistency", False),
    "neg_mean_logprob": ("Mean log-prob", "likelihood", False),
    "neg_sum_logprob": ("Sum log-prob", "likelihood", False),
    "neg_min_logprob": ("Min token log-prob", "likelihood", False),
    "mean_entropy": ("Mean token entropy", "likelihood", False),
    "max_entropy": ("Max token entropy", "likelihood", False),
    "neg_p_true": ("P(True)", "self-eval", False),
    "length": ("Answer length", "baseline", False),
    "probe_answer": ("Answer probe", "probe", False),
    "probe_centered": ("Centred probe", "probe", False),
    "probe_twoblock": ("Two-block probe", "probe", False),
}


def run_metrics(df: pd.DataFrame, det: str, n_boot: int) -> dict:
    s_all = df[det].to_numpy()
    samp = df[df.a > 0]
    s, y, g = samp[det].to_numpy(), samp.y.to_numpy(), samp.q.to_numpy()
    out = decompose(s, y, g)
    if n_boot:
        bs = bootstrap_decompose(s, y, g, n_boot=n_boot)
        for k in ["auroc", "auroc_w", "auroc_b", "auroc_w_macro"]:
            v = np.array([b[k] for b in bs])
            out[k + "_lo"], out[k + "_hi"] = np.nanpercentile(v, [2.5, 97.5])
    bon = best_of_n(s, 1 - y, g)
    out.update(bon)
    gr = df[df.a == 0]
    gd = decompose(gr[det].to_numpy(), gr.y.to_numpy(), gr.q.to_numpy())
    out["abst_auroc"] = gd["auroc"]
    out["abst_aurc"] = aurc(gr[det].to_numpy(), gr.y.to_numpy())
    return out


def majority_labels(df: pd.DataFrame) -> pd.Series:
    """Per mixed question: is the plurality cluster among samples correct, wrong, or tied?"""
    samp = df[df.a > 0]
    lab = {}
    for q, sub in samp.groupby("q"):
        k = sub.y.sum()
        if k == 0 or k == len(sub):
            continue
        counts = sub.cluster.value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            lab[q] = "tie"
            continue
        top = counts.index[0]
        lab[q] = "majority_correct" if sub[sub.cluster == top].y.mean() < 0.5 else "majority_wrong"
    return samp.q.map(lab)


def majority_split(df: pd.DataFrame, det: str, split: pd.Series) -> dict:
    """Within-question AUROC on mixed questions where the plurality cluster is correct vs wrong."""
    samp = df[df.a > 0].assign(split=split)
    res = {}
    for name in ["majority_correct", "majority_wrong"]:
        sub = samp[samp.split == name]
        if len(sub) == 0:
            res[name] = np.nan
            res["n_" + name] = 0
            continue
        d = decompose(sub[det].to_numpy(), sub.y.to_numpy(), sub.q.to_numpy())
        res[name] = d["auroc_w"]
        res["n_" + name] = d["n_mixed"]
        b = best_of_n(sub[det].to_numpy(), 1 - sub.y.to_numpy(), sub.q.to_numpy())
        res["sel_" + name] = b["sel_acc"]
        res["rand_" + name] = b["rand_acc"]
    return res


def analyse_run(path: str, n_boot: int):
    """All metrics for one run (one worker process)."""
    from pathlib import Path
    p = Path(path)
    model, dataset = p.parent.name.split("__")
    df = pd.read_parquet(p)
    samp = df[df.a > 0]
    per_q = samp.groupby("q").y.agg(["sum", "count"])
    mixed = ((per_q["sum"] > 0) & (per_q["sum"] < per_q["count"]))
    data_row = {
        "model": model, "dataset": dataset, "n_q": df.q.nunique(),
        "acc_greedy": 1 - df[df.a == 0].y.mean(), "acc_sample": 1 - samp.y.mean(),
        "frac_mixed": mixed.mean(), "frac_all_correct": (per_q["sum"] == 0).mean(),
        "frac_all_wrong": (per_q["sum"] == per_q["count"]).mean(),
        "judge_alias_agree": (df.y_judge == df.y_alias).mean(),
        "pi_w": decompose(samp.neg_mean_logprob, samp.y, samp.q)["pi_w"],
    }
    split = majority_labels(df)
    met, maj = [], []
    for det in DETECTORS:
        met.append({"model": model, "dataset": dataset, "detector": det, **run_metrics(df, det, n_boot)})
        maj.append({"model": model, "dataset": dataset, "detector": det, **majority_split(df, det, split)})
    return data_row, met, maj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    res_dir = OUT / ("results" + args.suffix)
    res_dir.mkdir(exist_ok=True)
    runs = sorted(p.name for p in OUT.iterdir() if (p / f"items{args.suffix}.parquet").exists())
    data_rows, met_rows, maj_rows = [], [], []
    paths = [str(OUT / run / f"items{args.suffix}.parquet") for run in runs]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(paths))) as ex:
        for run, (d, m, j) in zip(runs, ex.map(analyse_run, paths, [args.boot] * len(paths))):
            data_rows.append(d)
            met_rows += m
            maj_rows += j
            print("analysed", run, flush=True)
    data = pd.DataFrame(data_rows)
    met = pd.DataFrame(met_rows)
    maj = pd.DataFrame(maj_rows)
    data.to_csv(res_dir / "data_summary.csv", index=False)
    met.to_csv(res_dir / "metrics_per_run.csv", index=False)
    maj.to_csv(res_dir / "majority_split.csv", index=False)

    agg = met.groupby("detector")[["auroc", "auroc_b", "auroc_w", "auroc_w_macro", "gain_mixed",
                                   "sel_acc", "abst_auroc", "abst_aurc"]].mean()
    agg["n_runs_w_gt_half"] = met.assign(sig=met.get("auroc_w_lo", np.nan) > 0.5).groupby("detector").sig.sum()
    agg = agg.loc[[d for d in DETECTORS if d in agg.index]]
    agg.insert(0, "name", [DETECTORS[d][0] for d in agg.index])
    agg.to_csv(res_dir / "metrics_mean.csv")
    majagg = maj.groupby("detector")[["majority_correct", "majority_wrong"]].mean()
    # selection accuracy on the split, pooled over runs (weighted by number of questions)
    for name in ["majority_correct", "majority_wrong"]:
        w = maj["n_" + name]
        for k in ["sel_", "rand_"]:
            majagg[k + name] = (maj[k + name] * w).groupby(maj.detector).sum() / w.groupby(maj.detector).sum()
    majagg.to_csv(res_dir / "majority_split_mean.csv")

    # predictive validity across (run, answer-level detector) cells
    cells = met[~met.detector.map(lambda d: DETECTORS[d][2])]
    rows = []
    for target, preds in [("gain_mixed", ["auroc", "auroc_b", "auroc_w", "auroc_w_macro"]),
                          ("abst_auroc", ["auroc", "auroc_b", "auroc_w"])]:
        for p in preds:
            rho, pv = spearmanr(cells[p], cells[target])
            # within-run correlation (removes run-level offsets)
            per_run = [spearmanr(g[p], g[target])[0] for _, g in cells.groupby(["model", "dataset"])]
            rows.append({"target": target, "predictor": p, "spearman_all": rho, "p": pv,
                         "spearman_within_run_mean": np.nanmean(per_run)})
    pd.DataFrame(rows).to_csv(res_dir / "predictive_validity.csv", index=False)

    pd.set_option("display.width", 200)
    print(data.round(3).to_string())
    print(agg.round(3).to_string())
    print(majagg.round(3).to_string())
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
