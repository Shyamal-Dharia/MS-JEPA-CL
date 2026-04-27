import pandas as pd
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "preprocessed_data_fir" / "full"

def process_metadata(file_path, site_col, age_col, sex_col, dataset_name):
    df = pd.read_csv(file_path)
    if age_col not in df.columns or sex_col not in df.columns:
        print(f"Required columns missing in {dataset_name}.")
        return

    df = df.dropna(subset=[site_col, age_col, sex_col])

    # Convert sex to str to inspect
    df[sex_col] = df[sex_col].astype(str).str.upper()
    df["is_male"] = df[sex_col].apply(lambda x: 1 if x in ["1", "1.0", "M", "MALE", "B"] else 0)
    df["is_female"] = df[sex_col].apply(lambda x: 1 if x in ["2", "2.0", "F", "FEMALE", "G"] else 0)

    # Some datasets use 1/2 for M/F, others use M/F

    stats = df.groupby(site_col).agg(
        n_sub=(age_col, "count"),
        mean_age=(age_col, "mean"),
        n_male=("is_male", "sum"),
        n_female=("is_female", "sum")
    ).reset_index()

    stats["mean_age"] = stats["mean_age"].round(2)
    stats["n_male"] = stats["n_male"].astype(int)
    stats["n_female"] = stats["n_female"].astype(int)

    print(f"% --- {dataset_name} ---")
    print(r"\begin{tabular}[t]{lrr|r}")
    print(r"\toprule")
    print(rf"\multicolumn{{4}}{{c}}{{\textbf{{{dataset_name}}}}}\\")
    print(r"\cmidrule(lr){1-4}")
    print(r"\textbf{Site Name} & \textbf{N Subjects} & \textbf{Mean Age} & \textbf{Sex (\%M)} \\")
    print(r"\midrule")
    for _, row in stats.iterrows():
        site = str(row[site_col]).replace("ABIDEII-", "").replace("_1", "")  # cleanup ABIDE-II names a bit but left original earlier
        site = site.replace("_Long", "").replace("ABIDEII-", "")
        # to match previous formatting cleanly we use the row[site_col] but maybe replace
        actual_site = str(row[site_col])
        if "ABIDEII" in actual_site: actual_site = actual_site.replace("ABIDEII-", "")

        n_sub = row["n_sub"]
        mean_a = row["mean_age"]
        n_m = row["n_male"]
        n_f = row["n_female"]
        pct_m = int(round(n_m / n_sub * 100)) if n_sub > 0 else 0
        print(f"{actual_site} & {n_sub} & {mean_a} & {pct_m}\\% \\\\")
    print(r"\midrule")
    total_count = stats["n_sub"].sum()
    mean_all = df[age_col].mean()
    tot_m = stats["n_male"].sum()
    tot_pct_m = int(round(tot_m / total_count * 100)) if total_count > 0 else 0
    print(f"\\textbf{{Total}} & \\textbf{{{total_count}}} & \\textbf{{{mean_all:.2f}}} & \\textbf{{{tot_pct_m}\\%}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("\n")

process_metadata(DATA_ROOT / "subject_metadata_full.csv", "site", "age", "sex", "SRPBS-1600")
process_metadata(DATA_ROOT / "abide_metadata_full.csv", "site", "age", "sex", "ABIDE-I")
process_metadata(DATA_ROOT / "abide2_metadata.csv", "site", "age", "sex", "ABIDE-II")
