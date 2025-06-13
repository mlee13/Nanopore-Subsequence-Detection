import pandas as pd
import numpy as np
from pathlib import Path
from statistics import mode, StatisticsError

def analyse_results_csv(path):
    """
    Create an *_analysis.csv file with summary stats
    for each dataset block in the input CSV.
    """
    path = Path(path)
    out_path = path.with_name(path.stem + "_analysis.csv")

    # --- read raw lines so we can keep the dataset markers -------------
    with path.open() as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    header = lines[0].split(",")
    blocks = {}
    current_name = "UNNAMED"

    # --- split into blocks --------------------------------------------
    rows = []
    for ln in lines[1:]:
        if ln.startswith(">"):
            current_name = ln[1:].strip() or "UNNAMED"
            continue
        if not ln.strip():
            continue
        blocks.setdefault(current_name, []).append(ln.split(","))

    # --- helper for safe mode -----------------------------------------
    def safe_mode(series):
        try:
            return mode(series)
        except StatisticsError:
            return "NA"

    # --- accumulate per-dataset stats ---------------------------------
    records = []
    for name, rows in blocks.items():
        df = pd.DataFrame(rows, columns=header)

        df["TargetLength"] = pd.to_numeric(df["TargetLength"])
        scores = ["EucScore", "NCCScore", "DTWScore"]
        for s in scores:
            df[s] = pd.to_numeric(df[s])

        # basic counts
        n_targets = len(df)
        label_counts = df["ActualLabel"].value_counts()
        totals = {lab: label_counts.get(lab, 0) for lab in ["match", "non-match", "undefined"]}
        perc   = {f"perc_{k}": 100 * v / n_targets for k, v in totals.items()}

        # helper for descriptive stats
        def desc(col):
            return {
                f"avg_{col}": df[col].mean(),
                f"med_{col}": df[col].median(),
                f"range_{col}": df[col].max() - df[col].min(),
                f"mode_{col}": safe_mode(df[col])
            }

        # collect stats
        rec = {"Dataset": name, "n_targets": n_targets}
        rec.update(desc("TargetLength"))
        for s in scores:
            rec.update(desc(s))

        # ExpectedLabel (could be mixed)
        rec["Expected_labels"] = ",".join(sorted(df["ExpectedLabel"].unique()))

        # counts + percentages
        rec.update(totals)
        rec.update(perc)

        # DTW / Euc when prediction correct
        correct = df[df["ActualLabel"] == df["ExpectedLabel"]]
        if not correct.empty:
            rec.update({
                "avg_DTW_score_when_correct": correct["DTWScore"].mean(),
                "med_DTW_score_when_correct": correct["DTWScore"].median(),
                "avg_NCC_score_when_correct": correct["NCCScore"].mean(),
                "med_NCC_score_when_correct": correct["NCCScore"].median(),
            })
        else:
            rec.update({
                "avg_DTW_score_when_correct": np.nan,
                "med_DTW_score_when_correct": np.nan,
                "avg_NCC_score_when_correct": np.nan,
                "med_NCC_score_when_correct": np.nan,
            })

        records.append(rec)

    # --- write aggregated CSV -----------------------------------------
    summary = pd.DataFrame(records)
    summary.to_csv(out_path, index=False)
    print(f"Analysis written to {out_path}")


if __name__ == "__main__":
    analyse_results_csv("tempFYP/test_main/results/long_random_10000_match_scores.csv")
