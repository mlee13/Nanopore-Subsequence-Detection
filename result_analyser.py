import re
from pathlib import Path
from statistics import mode, StatisticsError

import numpy as np
import pandas as pd


FLOAT_FMT = "{:.4f}".format       # 4-d.p. everywhere

results_csv = "tempFYP/test_main/results/long_random_10000_scores.csv"

# ----------------------------------------------------------------------
# 1. Core analysis
# ----------------------------------------------------------------------
def analyse_match_results(csv_path: str | Path):
    """
    Create two CSVs:
    1) <orig>_analysis.csv       – one row per test-signal × ExpectedLabel
    2) test_signals_analysis.csv – one row per test-signal (confusion matrix)
    """
    in_path = Path(csv_path)

    # -----------------------------------------------------------------
    # PRE-PARSE so '>alias' overrides the signal name
    # -----------------------------------------------------------------
    with open(in_path) as fh:
        raw_lines = [ln.rstrip("\n") for ln in fh]

    header = raw_lines[0].split(",")
    rows   = []
    current_alias = None

    for ln in raw_lines[1:]:
        if ln.startswith(">"):
            current_alias = ln[1:].strip() or None
            continue
        if not ln.strip():
            continue

        row = ln.split(",")
        #  position 1 in the header is "Filename"
        #  if alias provided, replace the filename's signal name
        if current_alias:
            # clone row so we don't mutate list re-use
            row = row[:]                 
            fn = row[header.index("Filename")]
            # replace the leading signal part with alias
            row[header.index("Filename")] = re.sub(
                r"^(.+?)_target_", f"{current_alias}_target_", fn
            )
        rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    df["ActualLabel"]   = df["ActualLabel"].astype(str).str.strip().str.lower()
    df["ExpectedLabel"] = df["ExpectedLabel"].astype(str).str.strip().str.lower()

    # --- extract test-signal name from Filename -----------------------
    r = re.compile(r"(?P<signal>.+?)_target_.+\.fa$")
    df["TestSignal"] = df["Filename"].str.extract(r, expand=False)

    num_cols = ["TargetLength", "EucScore", "NCCScore", "DTWScore", "dtwBestDist"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # -----------------------------------------------------------------
    # A. detailed set-by-set stats  (test-signal × ExpectedLabel)
    # -----------------------------------------------------------------
    records_a = []
    grp_a = df.groupby(["TestSignal", "ExpectedLabel"])
    for (signal, exp_lab), sub in grp_a:
        rec = {
            "DatasetName": signal,
            "ExpectedLabel": exp_lab,
            "n_targets": len(sub),
            "match":      (sub["ActualLabel"] == "match").sum(),
            "non-match":  (sub["ActualLabel"] == "non-match").sum(),
            "undefined":  (sub["ActualLabel"] == "undefined").sum(),
            # target length
            "avg_TLen": sub["TargetLength"].mean(),
            "med_TLen": sub["TargetLength"].median(),
            "range_TLen": sub["TargetLength"].max() - sub["TargetLength"].min(),
        }
        # per-score stats
        for col in ["EucScore", "NCCScore", "DTWScore"]:
            rec[f"avg_{col}"] = sub[col].mean()
            rec[f"med_{col}"] = sub[col].median()
            rec[f"std_{col}"] = sub[col].std(ddof=0)
        # median scores when prediction correct
        corr = sub[sub["ActualLabel"] == sub["ExpectedLabel"]]
        for col in ["EucScore", "NCCScore", "DTWScore"]:
            rec[f"med_{col}_correct"] = corr[col].median() if not corr.empty else np.nan
        
        # --- quartiles for rows where ActualLabel == 'undefined' ------------
        undef = sub[sub["ActualLabel"] == "undefined"]
        for col in ["EucScore", "NCCScore", "DTWScore"]:
            if undef.empty:
                rec[f"q1_{col}_undef"]  = np.nan
                rec[f"med_{col}_undef"] = np.nan
                rec[f"q3_{col}_undef"]  = np.nan
            else:
                rec[f"q1_{col}_undef"]  = np.percentile(undef[col], 25)
                rec[f"med_{col}_undef"] = np.percentile(undef[col], 50)
                rec[f"q3_{col}_undef"]  = np.percentile(undef[col], 75)
        # --------------------------------------------------------------------

        for col in ["dtwBestDist"]:
            rec[f"avg_{col}"] = sub[col].mean()

        # mode of TargetLength
        # rec["mode_TLen"] = safe_mode(sub["TargetLength"])
        records_a.append(rec)

    out_a = Path("tempFYP/test_main/results_analysis") / f"{in_path.stem}_analysis.csv"
    new_df_a = pd.DataFrame(records_a).round(4)

    out_a.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_a.exists()
    new_df_a.to_csv(
        out_a,
        mode="a",                     # append
        header=not file_exists,       # write header only the first time
        index=False,
        float_format="%.4f"
    )
    print(f"[+] per-set analysis  → {out_a}")

    # -----------------------------------------------------------------
    # B. signal-level confusion-matrix stats
    # -----------------------------------------------------------------
    records_b = []
    grp_b = df.groupby("TestSignal")
    for signal, sub in grp_b:
        # confusion counts (exclude undefined from TP/FP/FN/TN maths)
        sub_clean = sub[sub["ActualLabel"] != "undefined"]
        TP = ((sub_clean["ExpectedLabel"] == "match") &
              (sub_clean["ActualLabel"]   == "match")).sum()
        FN = ((sub_clean["ExpectedLabel"] == "match") &
              (sub_clean["ActualLabel"]   == "non-match")).sum()
        FP = ((sub_clean["ExpectedLabel"] == "non-match") &
              (sub_clean["ActualLabel"]   == "match")).sum()
        TN = ((sub_clean["ExpectedLabel"] == "non-match") &
              (sub_clean["ActualLabel"]   == "non-match")).sum()

        tot_clean = TP + FP + FN + TN
        undef_cnt = (sub["ActualLabel"] == "undefined").sum()
        undef_rate = undef_cnt / len(sub) if len(sub) else np.nan

        rec = {
            "TestSignal": signal,
            "n_match_tests": (sub["ExpectedLabel"] == "match").sum(),
            "n_nonmatch_tests": (sub["ExpectedLabel"] == "non-match").sum(),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "Accuracy": (TP + TN) / tot_clean if tot_clean else np.nan,
            "Recall":   TP / (TP + FN)        if (TP + FN) else np.nan,
            "Precision":TP / (TP + FP)        if (TP + FP) else np.nan,
            "FNR":      FN / (TP + FN)        if (TP + FN) else np.nan,
            "F1": (2*TP) / (2*TP + FP + FN) if (2*TP + FP + FN) else np.nan,
            "UndefinedRate": undef_rate,
        }
        records_b.append(rec)

    # -----------------------------------------------------------------
    #  append results
    # -----------------------------------------------------------------
    out_b = Path("tempFYP/test_main/results_analysis/test_signals_analysis.csv")
    new_df_b = pd.DataFrame(records_b).round(4)

    out_b.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_b.exists()
    new_df_b.to_csv(
        out_b,
        mode="a",                     # append
        header=not file_exists,       # write header only the first time
        index=False,
        float_format="%.4f"
    )
    print(f"[+] test-signal analysis {'APPENDED' if file_exists else 'CREATED'} → {out_b}")

    # out_b = in_path.with_name("test_signals_analysis.csv")
    # pd.DataFrame(records_b).round(4).to_csv(out_b, index=False, float_format="%.4f")
    # print(f"[+] test-signal analysis → {out_b}")


# ----------------------------------------------------------------------
# 3.  Quick pretty-print helper
# ----------------------------------------------------------------------
def pretty_print_csv(csv_path: str | Path, max_rows=20):
    """
    Print a CSV in a readable table with floats rounded to 4 d.p.
    """
    import textwrap
    df = pd.read_csv(csv_path)
    pd.set_option("display.float_format", FLOAT_FMT)
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)-max_rows} more rows)")
    else:
        print(df.to_string(index=False))
    pd.reset_option("display.float_format")

# ----------------------------------------------------------------------
# 4.  If run as script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # print("You can use script by: python results_analysis.py <results.csv>")
        analyse_match_results(results_csv)
    else:
        analyse_match_results(sys.argv[1])
