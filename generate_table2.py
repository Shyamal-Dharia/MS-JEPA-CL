import numpy as np
import pandas as pd
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "preprocessed_data_fir" / "full"

def process_srpbs(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["site", "age", "sex", "diag_name"])
    df = df[df["age"] > 0]

    df["dx_std"] = df["diag_name"].str.upper()
    df["label"] = np.where(df["dx_std"].str.contains("HEALTHY") | df["dx_std"].str.contains("CONTROL"), 0, 1)

    site_counts = df.groupby(["site", "label"]).size().unstack(fill_value=0)
    valid_sites = site_counts[(site_counts[0] > 0) & (site_counts[1] > 0)].index
    df = df[df["site"].isin(valid_sites)]

    n = len(df)
    sites = df["site"].nunique()
    age_mean = df["age"].mean()
    age_std = df["age"].std()

    df["sex"] = df["sex"].astype(str).str.upper()
    m_count = len(df[df["sex"].isin(["1", "1.0", "M", "MALE", "B"])])
    f_count = len(df[df["sex"].isin(["2", "2.0", "F", "FEMALE", "G"])])

    hc = len(df[df["label"] == 0])
    asd = len(df[df["dx_std"].str.contains("AUTISM") | df["dx_std"].str.contains("ASD")])
    mdd = len(df[df["dx_std"].str.contains("DEPRESS") | df["dx_std"].str.contains("MDD")])
    scz = len(df[df["dx_std"].str.contains("SCHIZO") | df["dx_std"].str.contains("SCZ")])
    bd = len(df[df["dx_std"].str.contains("BIPOLAR") | df["dx_std"].str.contains("BD")])
    pain = len(df[df["dx_std"].str.contains("PAIN")])
    stroke = len(df[df["dx_std"].str.contains("STROKE")])
    dys = len(df[df["dx_std"].str.contains("DYS") | df["dx_std"].str.contains("DYSTHYM")])
    accounted = hc + asd + mdd + scz + bd + pain + stroke + dys
    oth = n - accounted

    return {
        "Dataset": "SRPBS", "N": n, "Sites": sites, "Age_mean": age_mean, "Age_std": age_std,
        "M": m_count, "F": f_count,
        "HC": hc, "ASD": asd, "MDD": mdd, "SCZ": scz, "BD": bd, "Pain": pain, "Stroke": stroke, "Dys": dys, "Oth": oth
    }

def process_abide(path, name):
    df = pd.read_csv(path)
    if "SITE_ID" in df.columns: df = df.rename(columns={"SITE_ID": "site"})
    if "AGE_AT_SCAN" in df.columns: df = df.rename(columns={"AGE_AT_SCAN": "age"})
    if "SEX" in df.columns: df = df.rename(columns={"SEX": "sex"})

    df = df.dropna(subset=["site", "age", "sex", "dx_name"])
    df = df[df["age"] > 0]

    df["dx_std"] = df["dx_name"].str.upper()
    df = df[df["dx_std"].str.contains("AUTISM") | df["dx_std"].str.contains("CONTROL")]
    df["label"] = np.where(df["dx_std"].str.contains("CONTROL"), 0, 1)

    site_counts = df.groupby(["site", "label"]).size().unstack(fill_value=0)
    valid_sites = site_counts[(site_counts[0] > 0) & (site_counts[1] > 0)].index
    df = df[df["site"].isin(valid_sites)]

    n = len(df)
    sites = df["site"].nunique()
    age_mean = df["age"].mean()
    age_std = df["age"].std()

    df["sex"] = df["sex"].astype(str).str.upper()
    m_count = len(df[df["sex"].isin(["1", "1.0", "M", "MALE", "B"])])
    f_count = len(df[df["sex"].isin(["2", "2.0", "F", "FEMALE", "G"])])

    hc = len(df[df["label"] == 0])
    asd = len(df[df["label"] == 1])

    return {
        "Dataset": name, "N": n, "Sites": sites, "Age_mean": age_mean, "Age_std": age_std,
        "M": m_count, "F": f_count,
        "HC": hc, "ASD": asd, "MDD": 0, "SCZ": 0, "BD": 0, "Pain": 0, "Stroke": 0, "Dys": 0, "Oth": 0
    }

d1 = process_srpbs(DATA_ROOT / "subject_metadata_full.csv")
d2 = process_abide(DATA_ROOT / "abide_metadata_full.csv", "ABIDE-I")
d3 = process_abide(DATA_ROOT / "abide2_metadata.csv", "ABIDE-II")

all_d = [d1, d2, d3]
d_tot = {
    "Dataset": "Total",
    "N": sum(d["N"] for d in all_d),
    "Sites": sum(d["Sites"] for d in all_d),
    "M": sum(d["M"] for d in all_d),
    "F": sum(d["F"] for d in all_d),
    "HC": sum(d["HC"] for d in all_d),
    "ASD": sum(d["ASD"] for d in all_d),
    "MDD": sum(d["MDD"] for d in all_d),
    "SCZ": sum(d["SCZ"] for d in all_d),
    "BD": sum(d["BD"] for d in all_d),
    "Pain": sum(d["Pain"] for d in all_d),
    "Stroke": sum(d["Stroke"] for d in all_d),
    "Dys": sum(d["Dys"] for d in all_d),
    "Oth": sum(d["Oth"] for d in all_d),
}

age_sum = sum(d["Age_mean"] * d["N"] for d in all_d)
d_tot["Age_mean"] = age_sum / d_tot["N"]
ss_total = sum((d["N"]-1)*d["Age_std"]**2 + d["N"]*(d["Age_mean"] - d_tot["Age_mean"])**2 for d in all_d)
d_tot["Age_std"] = (ss_total / (d_tot["N"] - 1)) ** 0.5
all_d.append(d_tot)

print(r"\begin{table*}[t]")
print(r"\centering")
print(r"\caption{Subject-level summary of the datasets used in this study. Age is reported as mean $\pm$ standard deviation.}")
print(r"\label{tab:dataset_summary}")
print(r"\scriptsize")
print(r"\setlength{\tabcolsep}{4pt}")
print(r"\renewcommand{\arraystretch}{1.08}")
print(r"\resizebox{\textwidth}{!}{%")
print(r"\begin{tabular}{lccccccccccccc}")
print(r"\toprule")
print(r"& \multicolumn{4}{c}{Demographics} & \multicolumn{9}{c}{Class Distribution} \\")
print(r"\cmidrule(lr){2-5} \cmidrule(lr){6-14}")
print(r"Dataset & N & Sites & Age (years) & M/F & HC & ASD & MDD & SCZ & BD & Pain & Stroke & Dys. & Oth. \\")
print(r"\midrule")
for i, d in enumerate(all_d):
    if i == 3: print(r"\midrule")
    # escape the pm explicitly
    # print string
    ds = d['Dataset']
    n = d['N']
    sit = d['Sites']
    a_m = f"{d['Age_mean']:.2f}"
    a_s = f"{d['Age_std']:.2f}"
    mf = f"{d['M']} / {d['F']}"
    hc = d['HC']
    asd = d['ASD']
    mdd = d['MDD']
    scz = d['SCZ']
    bd = d['BD']
    pain = d['Pain']
    strok = d['Stroke']
    dys = d['Dys']
    oth = d['Oth']

    print(f"{ds:8} & {n} & {sit} & {a_m} \\pm {a_s} & {mf:10} & {hc} & {asd} & {mdd} & {scz} & {bd} & {pain} & {strok} & {dys} & {oth} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}%")
print(r"}")
print(r"\end{table*}")
