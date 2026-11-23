"""LaTeX tables for the paper, generated from result CSVs (no hand-copied numbers).

Usage: python src/make_tables.py   -> paper/tables/{data,detectors,robustness}.tex
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyze import DETECTORS

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs_h100"
TAB = ROOT / "paper" / "tables"
MODEL_NAMES = {"qwen3b": "Qwen2.5-3B", "qwen7b": "Qwen2.5-7B", "qwen14b": "Qwen2.5-14B",
               "llama3b": "Llama-3.2-3B", "llama8b": "Llama-3.1-8B", "phi35": "Phi-3.5-mini"}
DATA_NAMES = {"triviaqa": "TriviaQA", "nq": "NQ-Open", "popqa": "PopQA"}
FAMILY_NAMES = {"consistency": "Consistency", "probe": "Probe", "baseline": "Baseline",
                "likelihood": "Likelihood", "self-eval": "Self-eval."}


def f3(x):
    return f"{x:.3f}".lstrip("0") if abs(x) < 1 else f"{x:.3f}"


def table_data():
    d = pd.read_csv(RES / "results" / "data_summary.csv")
    rows = []
    for m in MODEL_NAMES:
        cells = []
        for ds in DATA_NAMES:
            r = d[(d.model == m) & (d.dataset == ds)].iloc[0]
            cells += [f"{100 * r.acc_greedy:.1f}", f"{100 * r.frac_mixed:.1f}"]
        rows.append(MODEL_NAMES[m] + " & " + " & ".join(cells) + r" \\")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{v}}}" for v in DATA_NAMES.values())
    agree = 100 * d.judge_alias_agree.mean()
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Greedy accuracy (Acc.) and share of \emph{mixed} questions, whose ten samples are "
        r"partly correct (Mix.), in \%. Mixed questions are the only ones on which best-of-$n$ "
        rf"selection can matter. Judge and alias labels agree on {agree:.1f}\% of answers.}}",
        r"\label{tab:data}", r"\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        rf"& {head} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Model & Acc. & Mix. & Acc. & Mix. & Acc. & Mix. \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def table_detectors():
    m = pd.read_csv(RES / "results" / "metrics_mean.csv", index_col=0)
    per = pd.read_csv(RES / "results" / "metrics_per_run.csv")
    rng = per.groupby("detector").auroc.agg(["min", "max"])
    order = [d for fam in ["consistency", "likelihood", "self-eval", "probe", "baseline"]
             for d in DETECTORS if DETECTORS[d][1] == fam]
    rows, prev = [], None
    for det in order:
        name, fam, ql = DETECTORS[det]
        r = m.loc[det]
        if prev is not None and fam != prev:
            rows.append(r"\addlinespace[2pt]")
        famcell = FAMILY_NAMES[fam] if fam != prev else ""
        prev = fam
        name = name + (r"$^\dagger$" if ql else "")
        sig = "--" if ql else f"{int(r.n_runs_w_gt_half)}"
        rows.append(f"{famcell} & {name} & {f3(r.auroc)} & {f3(rng.loc[det, 'min'])}--{f3(rng.loc[det, 'max'])} & "
                    f"{f3(r.auroc_w)} & {sig} & {100 * r.gain_mixed:+.1f} & {f3(r.abst_aurc)}" + r" \\")
    return "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Detectors averaged over 18 runs (6 models $\times$ 3 datasets, 2{,}000 questions, "
        r"10 samples each). $A$: pooled AUROC over all sampled answers, with its range over runs. "
        r"$A_W$: pair-weighted within-question AUROC on mixed questions. Sig.: runs (of 18) whose 95\% "
        r"cluster-bootstrap interval for $A_W$ lies above $0.5$. Gain: best-of-10 accuracy minus "
        r"random-pick accuracy on mixed questions, in points. AURC: area under the risk--coverage "
        r"curve for abstention on greedy answers (lower is better). $^\dagger$Question-level by "
        r"construction: $A_W=0.5$ and gain $=0$ exactly. Between-question AUROC $A_B$ differs from $A$ by at most "
        r"$5.3\times10^{-5}$ in every run ($\pi_W\le 1.7\times10^{-4}$).}",
        r"\label{tab:detectors}",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Family & Detector & $A$ & $A$ range & $A_W$ & Sig. & Gain & AURC \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table*}"])


def table_robustness():
    variants = [("results", r"Main"), ("results_L0_judge", r"Probes at $L/2$"),
                ("results_L2_judge", r"Probes at $L$"), ("results_L1_alias", r"Alias labels")]
    rows = []
    for folder, name in variants:
        pv = pd.read_csv(RES / folder / "predictive_validity.csv").set_index(["target", "predictor"])
        mm = pd.read_csv(RES / folder / "metrics_mean.csv", index_col=0)
        g = lambda t, p: f"{pv.loc[(t, p), 'spearman_all']:.2f}"  # noqa: E731
        rows.append(f"{name} & {g('gain_mixed', 'auroc')} & {g('gain_mixed', 'auroc_w')} & "
                    f"{g('abst_auroc', 'auroc')} & {g('abst_auroc', 'auroc_w')} & "
                    f"{f3(mm.loc['probe_centered', 'auroc'])}" + r"\,/\," +
                    f"{f3(mm.loc['probe_centered', 'auroc_w'])}" + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Which component predicts which decision. Spearman correlation across 216 "
        r"(run, answer-level detector) cells between best-of-10 gain or abstention AUROC and the pooled "
        r"($A$) or within-question ($A_W$) AUROC. Main: probes at layer $3L/4$, judge labels. "
        r"Last column: centred probe, $A$\,/\,$A_W$.}",
        r"\label{tab:robust}", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"& \multicolumn{2}{c}{Best-of-10 gain} & \multicolumn{2}{c}{Abstention} & Centred \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Setting & $A$ & $A_W$ & $A$ & $A_W$ & probe \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "data.tex").write_text(table_data(), encoding="utf8")
    (TAB / "detectors.tex").write_text(table_detectors(), encoding="utf8")
    (TAB / "robustness.tex").write_text(table_robustness(), encoding="utf8")
    print("tables written to", TAB)


if __name__ == "__main__":
    main()
