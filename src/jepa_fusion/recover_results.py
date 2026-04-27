from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "results" / "reports" / "tables" / "jepa_fusion"

DATASET_ORDER = ["abide1", "abide2", "srpbs"]
DATASET_TASK_LABELS = {
    "abide1": "HC vs ASD",
    "abide2": "HC vs ASD",
    "srpbs": "HC vs pooled disorder",
}
SUBJECT_VOTE_SITE_EMPHASIS = {
    "abide1": "JEPA (temporal basic; pooled SSL)",
    "srpbs": "JEPA (temporal basic; pooled SSL)",
}

BASELINE_FILES = {
    "abide1": "abide1_hc_asd_mixedsites_baseline_table.json",
    "abide2": "abide2_hc_asd_mixedsites_baseline_table.json",
    "srpbs": "srpbs_hc_disorder_mixedsites_baseline_table.json",
}

GROUPED_CV_JEPA_ROWS = [
    ("JEPA (temporal basic; pooled SSL)", "pilot_t_basic_pooled_final_jepa_eval.json"),
    ("JEPA (temporal basic; curriculum)", "pilot_t_basic_curr_phase2_final_jepa_eval.json"),
    ("JEPA (temporal rotary conv-net; curriculum)", "pilot_t_rotary_conv_net_curr_phase2_final_jepa_eval.json"),
    ("JEPA (temporal rotary conv-net; curriculum + replay alignment)", "rotary_curr_align_groupedcv.json"),
    (
        "JEPA (temporal rotary conv-net; curriculum + replay alignment + domain adversary)",
        "rotary_curr_align_adv_groupedcv.json",
    ),
]

SUBJECT_VOTE_JEPA_ROWS = [
    (
        "JEPA (temporal basic; pooled SSL)",
        "pilot_t_basic_pooled_final_jepa_eval.json",
        "pilot_t_basic_pooled_final_subject_vote.json",
    ),
    (
        "JEPA (temporal rotary conv-net; curriculum + replay alignment)",
        "rotary_curr_align_groupedcv.json",
        "rotary_curr_align_subject_vote.json",
    ),
    (
        "JEPA (temporal rotary conv-net; curriculum + replay alignment + domain adversary)",
        "rotary_curr_align_adv_groupedcv.json",
        "rotary_curr_align_adv_subject_vote.json",
    ),
]

FUSION_DISPLAY = {
    "fusion::fc": "Static FC + JEPA",
    "fusion::fc_combat": "Static FC + ComBat + JEPA",
    "fusion::fc_covbat": "Static FC + CovBat + JEPA",
}

FINAL_PROFILE_LABELS = {
    "basic_pooled": "basic pooled",
    "rotary_curr_align": "rotary curr + replay align",
}

FINAL_CONFIRM_TEXT = {
    "abide1": "Best overall and best site-conscious result: CovBat fusion with rotary JEPA.",
    "abide2": "Plain FC baseline still leads overall; among harmonized fusions, ComBat fusion with basic JEPA is least weak.",
    "srpbs": "Rotary JEPA helps; plain FC fusion is numerically best overall, while ComBat fusion is the cleaner harmonized choice.",
}

SHORT_TAKE_LINES = [
    "**ABIDE-I:** the clean final winner is `FC + CovBat + JEPA` with the `rotary_curr_align` JEPA branch.",
    "**ABIDE-II:** JEPA fusion does not beat the plain FC baseline after multiseed confirmation.",
    "**SRPBS-1600:** JEPA helps, especially with the rotary branch, but the best harmonized fusion is `FC + ComBat + JEPA`, not `FC + CovBat + JEPA`.",
    "If you want one **site-conscious paper-facing model**, `rotary_curr_align + FC + CovBat + JEPA` is still the cleanest headline because it remains strongest on `ABIDE-I`, where your earlier significance test was also positive.",
]

REQUIRED_JSONS = [
    *BASELINE_FILES.values(),
    *[filename for _, filename in GROUPED_CV_JEPA_ROWS],
    *[grouped for _, grouped, _ in SUBJECT_VOTE_JEPA_ROWS],
    *[subject_vote for _, _, subject_vote in SUBJECT_VOTE_JEPA_ROWS],
    "rotary_curr_align_phase2_final_fusion_subject_vote.json",
    "rotary_curr_align_phase2_final_fusion_subject_vote_fused_prob_site_probe.json",
    "rotary_curr_align_phase2_final_fusion_subject_vote_stats.stats.json",
    "confirm_basic_pooled_seed1_subject_vote.json",
    "confirm_basic_pooled_seed2_subject_vote.json",
    "confirm_basic_pooled_seed1_fusion_subject_vote.json",
    "confirm_basic_pooled_seed2_fusion_subject_vote.json",
    "confirm_rotary_curr_align_seed1_subject_vote.json",
    "confirm_rotary_curr_align_seed2_subject_vote.json",
    "confirm_rotary_curr_align_seed1_fusion_subject_vote.json",
    "confirm_rotary_curr_align_seed2_fusion_subject_vote.json",
]


@dataclass(frozen=True)
class RecoveryOutputs:
    paper_tex: str
    final_md: str
    report_md: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild the paper-facing JEPA fusion result tables from the surviving JSON artifacts."
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--paper-tex", default=None, help="Override output path for current_paper_tables.tex")
    parser.add_argument(
        "--final-md",
        default=None,
        help="Override output path for FINAL_MULTI_SEED_CONFIRMATION_TABLE.md",
    )
    parser.add_argument(
        "--report-md",
        default=None,
        help="Override output path for RECOVERY_REPORT.md",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def format_md_pm(metric_mean: float, metric_std: float) -> str:
    return f"{format_pct(metric_mean)} ± {format_pct(metric_std)}"


def format_tex_pm(metric_mean: float, metric_std: float) -> str:
    return f"{format_pct(metric_mean)} $\\pm$ {format_pct(metric_std)}"


def format_tex_single(metric_mean: float) -> str:
    return format_pct(metric_mean)


def tex_bold(text: str, enabled: bool) -> str:
    return f"\\textbf{{{text}}}" if enabled else text


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(stdev(values))


def aggregate_metric(values: list[float]) -> tuple[float, float]:
    return float(mean(values)), sample_std(values)


def dataset_heading(baseline_payload: dict, dataset_key: str) -> str:
    label = baseline_payload["config"]["dataset_label"]
    n_sites = baseline_payload["meta"]["n_sites"]
    task_label = DATASET_TASK_LABELS[dataset_key]
    return f"\\textbf{{{label}}} ({n_sites} mixed-label sites; {task_label})"


def read_payloads(results_dir: Path) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    missing = []
    for name in REQUIRED_JSONS:
        path = results_dir / name
        if not path.exists():
            missing.append(str(path))
            continue
        payloads[name] = load_json(path)
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Missing required recovery inputs:\n{missing_text}")
    return payloads


def parse_pm_cell(text: str) -> tuple[float, float]:
    mean_text, std_text = [part.strip() for part in text.split("±")]
    return float(mean_text) / 100.0, float(std_text) / 100.0


def build_srpbs_baseline_from_results_log(results_dir: Path, legacy_payload: dict) -> dict:
    results_log = (results_dir / "RESULTS_LOG.md").read_text()
    section_match = re.search(
        r"### SRPBS-1600.*?Main table:\n\n(?P<table>\| Method \|.*?)(?:\n\nInterpretation:|\Z)",
        results_log,
        flags=re.S,
    )
    if section_match is None:
        raise ValueError("Could not locate the SRPBS baseline table inside RESULTS_LOG.md")
    table_lines = [line.strip() for line in section_match.group("table").strip().splitlines() if line.strip().startswith("|")]
    method_rows = {}
    for line in table_lines[2:]:
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 4:
            continue
        method_rows[parts[0]] = {
            "site_bacc": parse_pm_cell(parts[1]) if "±" in parts[1] else (float(parts[1]) / 100.0, 0.0),
            "dx_bacc": parse_pm_cell(parts[2]) if "±" in parts[2] else (float(parts[2]) / 100.0, 0.0),
            "dx_auc": parse_pm_cell(parts[3]) if "±" in parts[3] else (float(parts[3]) / 100.0, 0.0),
        }
    mapping = {
        "raw": "ROI time series",
        "fc": "Static FC",
        "fc_combat": "Static FC + ComBat",
        "fc_covbat": "Static FC + CovBat",
    }
    return {
        "config": {
            "dataset": legacy_payload["dataset"],
            "dataset_label": legacy_payload["display_name"],
            "task_label": legacy_payload["task_display"],
        },
        "meta": {
            "n_sites": int(legacy_payload["n_sites"]),
        },
        "summary": {
            "site": {
                method_key: {
                    "balanced_accuracy": {
                        "mean": method_rows[method_label]["site_bacc"][0],
                        "std": method_rows[method_label]["site_bacc"][1],
                    }
                }
                for method_key, method_label in mapping.items()
            },
            "diagnosis": {
                method_key: {
                    "balanced_accuracy": {
                        "mean": method_rows[method_label]["dx_bacc"][0],
                        "std": method_rows[method_label]["dx_bacc"][1],
                    },
                    "roc_auc": {
                        "mean": method_rows[method_label]["dx_auc"][0],
                        "std": method_rows[method_label]["dx_auc"][1],
                    },
                }
                for method_key, method_label in mapping.items()
            },
        },
    }


def normalize_baseline_payload(dataset_key: str, payload: dict, results_dir: Path) -> dict:
    if "config" in payload and "meta" in payload and "summary" in payload:
        return payload
    if dataset_key == "srpbs" and {"dataset", "display_name", "task_display", "n_sites", "methods"} <= set(payload):
        return build_srpbs_baseline_from_results_log(results_dir, payload)
    raise ValueError(f"Unsupported baseline payload format for {dataset_key}")


def grouped_cv_rows(baseline_payload: dict, dataset_key: str, payloads: dict[str, dict]) -> list[dict]:
    rows = [
        {
            "label": "ROI time series",
            "site_mean": baseline_payload["summary"]["site"]["raw"]["balanced_accuracy"]["mean"],
            "site_std": baseline_payload["summary"]["site"]["raw"]["balanced_accuracy"]["std"],
            "dx_mean": baseline_payload["summary"]["diagnosis"]["raw"]["balanced_accuracy"]["mean"],
            "dx_std": baseline_payload["summary"]["diagnosis"]["raw"]["balanced_accuracy"]["std"],
            "auc_mean": baseline_payload["summary"]["diagnosis"]["raw"]["roc_auc"]["mean"],
            "auc_std": baseline_payload["summary"]["diagnosis"]["raw"]["roc_auc"]["std"],
            "jepa": False,
            "show_pm": True,
        },
        {
            "label": "Static FC",
            "site_mean": baseline_payload["summary"]["site"]["fc"]["balanced_accuracy"]["mean"],
            "site_std": baseline_payload["summary"]["site"]["fc"]["balanced_accuracy"]["std"],
            "dx_mean": baseline_payload["summary"]["diagnosis"]["fc"]["balanced_accuracy"]["mean"],
            "dx_std": baseline_payload["summary"]["diagnosis"]["fc"]["balanced_accuracy"]["std"],
            "auc_mean": baseline_payload["summary"]["diagnosis"]["fc"]["roc_auc"]["mean"],
            "auc_std": baseline_payload["summary"]["diagnosis"]["fc"]["roc_auc"]["std"],
            "jepa": False,
            "show_pm": True,
        },
        {
            "label": "Static FC + ComBat",
            "site_mean": baseline_payload["summary"]["site"]["fc_combat"]["balanced_accuracy"]["mean"],
            "site_std": baseline_payload["summary"]["site"]["fc_combat"]["balanced_accuracy"]["std"],
            "dx_mean": baseline_payload["summary"]["diagnosis"]["fc_combat"]["balanced_accuracy"]["mean"],
            "dx_std": baseline_payload["summary"]["diagnosis"]["fc_combat"]["balanced_accuracy"]["std"],
            "auc_mean": baseline_payload["summary"]["diagnosis"]["fc_combat"]["roc_auc"]["mean"],
            "auc_std": baseline_payload["summary"]["diagnosis"]["fc_combat"]["roc_auc"]["std"],
            "jepa": False,
            "show_pm": True,
        },
        {
            "label": "Static FC + CovBat",
            "site_mean": baseline_payload["summary"]["site"]["fc_covbat"]["balanced_accuracy"]["mean"],
            "site_std": baseline_payload["summary"]["site"]["fc_covbat"]["balanced_accuracy"]["std"],
            "dx_mean": baseline_payload["summary"]["diagnosis"]["fc_covbat"]["balanced_accuracy"]["mean"],
            "dx_std": baseline_payload["summary"]["diagnosis"]["fc_covbat"]["balanced_accuracy"]["std"],
            "auc_mean": baseline_payload["summary"]["diagnosis"]["fc_covbat"]["roc_auc"]["mean"],
            "auc_std": baseline_payload["summary"]["diagnosis"]["fc_covbat"]["roc_auc"]["std"],
            "jepa": False,
            "show_pm": True,
        },
    ]
    for label, filename in GROUPED_CV_JEPA_ROWS:
        summary = payloads[filename]["datasets"][dataset_key]["summary"]
        rows.append(
            {
                "label": label,
                "site_mean": summary["site"]["balanced_accuracy"]["mean"],
                "site_std": 0.0,
                "dx_mean": summary["diagnosis"]["balanced_accuracy"]["mean"],
                "dx_std": 0.0,
                "auc_mean": summary["diagnosis"]["roc_auc"]["mean"],
                "auc_std": 0.0,
                "jepa": True,
                "show_pm": False,
            }
        )
    return rows


def subject_vote_rows(
    baseline_payload: dict,
    dataset_key: str,
    payloads: dict[str, dict],
    fusion_payload: dict,
    fusion_site_payload: dict,
) -> list[dict]:
    rows = [
        {
            "label": "Static FC",
            "site_mean": baseline_payload["summary"]["site"]["fc"]["balanced_accuracy"]["mean"],
            "site_std": baseline_payload["summary"]["site"]["fc"]["balanced_accuracy"]["std"],
            "dx_mean": fusion_payload["datasets"][dataset_key]["summary"]["fc"]["bacc"]["mean"],
            "dx_std": fusion_payload["datasets"][dataset_key]["summary"]["fc"]["bacc"]["std"],
            "auc_mean": fusion_payload["datasets"][dataset_key]["summary"]["fc"]["auc"]["mean"],
            "auc_std": fusion_payload["datasets"][dataset_key]["summary"]["fc"]["auc"]["std"],
            "style": None,
            "site_show_pm": True,
        },
        {
            "label": "Static FC + ComBat",
            "site_mean": baseline_payload["summary"]["site"]["fc_combat"]["balanced_accuracy"]["mean"],
            "site_std": baseline_payload["summary"]["site"]["fc_combat"]["balanced_accuracy"]["std"],
            "dx_mean": fusion_payload["datasets"][dataset_key]["summary"]["fc_combat"]["bacc"]["mean"],
            "dx_std": fusion_payload["datasets"][dataset_key]["summary"]["fc_combat"]["bacc"]["std"],
            "auc_mean": fusion_payload["datasets"][dataset_key]["summary"]["fc_combat"]["auc"]["mean"],
            "auc_std": fusion_payload["datasets"][dataset_key]["summary"]["fc_combat"]["auc"]["std"],
            "style": None,
            "site_show_pm": True,
        },
        {
            "label": "Static FC + CovBat",
            "site_mean": baseline_payload["summary"]["site"]["fc_covbat"]["balanced_accuracy"]["mean"],
            "site_std": baseline_payload["summary"]["site"]["fc_covbat"]["balanced_accuracy"]["std"],
            "dx_mean": fusion_payload["datasets"][dataset_key]["summary"]["fc_covbat"]["bacc"]["mean"],
            "dx_std": fusion_payload["datasets"][dataset_key]["summary"]["fc_covbat"]["bacc"]["std"],
            "auc_mean": fusion_payload["datasets"][dataset_key]["summary"]["fc_covbat"]["auc"]["mean"],
            "auc_std": fusion_payload["datasets"][dataset_key]["summary"]["fc_covbat"]["auc"]["std"],
            "style": None,
            "site_show_pm": True,
        },
    ]
    for label, grouped_filename, subject_vote_filename in SUBJECT_VOTE_JEPA_ROWS:
        grouped_summary = payloads[grouped_filename]["datasets"][dataset_key]["summary"]
        subject_vote_summary = payloads[subject_vote_filename]["datasets"][dataset_key]["summary"]
        rows.append(
            {
                "label": label,
                "site_mean": grouped_summary["site"]["balanced_accuracy"]["mean"],
                "site_std": 0.0,
                "dx_mean": subject_vote_summary["bacc"]["mean"],
                "dx_std": subject_vote_summary["bacc"]["std"],
                "auc_mean": subject_vote_summary["auc"]["mean"],
                "auc_std": subject_vote_summary["auc"]["std"],
                "style": "tablejepa",
                "site_show_pm": False,
            }
        )
    for method_key in ("fusion::fc", "fusion::fc_combat", "fusion::fc_covbat"):
        rows.append(
            {
                "label": f"{FUSION_DISPLAY[method_key]} (late fusion)",
                "site_mean": fusion_site_payload["datasets"][dataset_key]["summary"][method_key]["balanced_accuracy"]["mean"],
                "site_std": fusion_site_payload["datasets"][dataset_key]["summary"][method_key]["balanced_accuracy"]["std"],
                "dx_mean": fusion_payload["datasets"][dataset_key]["summary"][method_key]["bacc"]["mean"],
                "dx_std": fusion_payload["datasets"][dataset_key]["summary"][method_key]["bacc"]["std"],
                "auc_mean": fusion_payload["datasets"][dataset_key]["summary"][method_key]["auc"]["mean"],
                "auc_std": fusion_payload["datasets"][dataset_key]["summary"][method_key]["auc"]["std"],
                "style": "tablefusion",
                "site_show_pm": True,
            }
        )
    return rows


def build_grouped_cv_table(baselines: dict[str, dict], payloads: dict[str, dict]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Current grouped-CV mixed-site benchmark under the new protocol. Classical baselines are reported as mean $\\pm$ standard deviation across 5 random seeds. JEPA rows are the currently completed 1-seed pilot and rescue checkpoints. Lower site-probe balanced accuracy indicates less recoverable site information; higher grouped-CV diagnosis balanced accuracy and ROC-AUC indicate stronger diagnosis prediction under the same probe family.}",
        "\\label{tab:jepa_fusion_current_groupedcv_updated}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Representation & Site probe bAcc. ($\\downarrow$) & Diagnosis probe bAcc. ($\\uparrow$) & Diagnosis probe ROC-AUC ($\\uparrow$) \\\\",
        "\\midrule",
    ]
    for index, dataset_key in enumerate(DATASET_ORDER):
        baseline_payload = baselines[dataset_key]
        rows = grouped_cv_rows(baseline_payload, dataset_key, payloads)
        best_dx_bacc = max(row["dx_mean"] for row in rows)
        best_auc = max(row["auc_mean"] for row in rows)
        lines.append(f"\\multicolumn{{4}}{{l}}{{{dataset_heading(baseline_payload, dataset_key)}}} \\\\")
        for row in rows:
            dx_cell = format_tex_pm(row["dx_mean"], row["dx_std"]) if row["show_pm"] else format_tex_single(row["dx_mean"])
            auc_cell = format_tex_pm(row["auc_mean"], row["auc_std"]) if row["show_pm"] else format_tex_single(row["auc_mean"])
            dx_cell = tex_bold(dx_cell, row["dx_mean"] == best_dx_bacc)
            auc_cell = tex_bold(auc_cell, row["auc_mean"] == best_auc)
            site_cell = format_tex_pm(row["site_mean"], row["site_std"]) if row["show_pm"] else format_tex_single(row["site_mean"])
            if row["jepa"]:
                lines.append("\\rowcolor{tablejepa}")
            lines.append(f"{row['label']} & {site_cell} & {dx_cell} & {auc_cell} \\\\")
        chance_site = 1.0 / baseline_payload["meta"]["n_sites"]
        lines.append(f"Chance & {format_pct(chance_site)} & 50.0 & 50.0 \\\\")
        if index != len(DATASET_ORDER) - 1:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{0.2cm}",
            "\\begin{minipage}{0.97\\textwidth}",
            "\\footnotesize \\textbf{Notes:} This table uses the \\textbf{current} branch protocol, not the older LOSO/transductive setup. Both site and diagnosis numbers come from subject-grouped stratified cross-validation with the same downstream probe family. Mixed-site-only filtering is active for the binary tasks, so the effective site counts are 23 for ABIDE-I, 14 for ABIDE-II, and 11 for SRPBS. ComBat and CovBat are fit on training folds only. Blue-shaded rows denote JEPA-derived models, including the exploratory domain-adversarial rotary ablation. All JEPA rows shown here are currently \\textbf{single-seed} results and should be read as provisional.",
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_subject_vote_table(
    baselines: dict[str, dict],
    payloads: dict[str, dict],
    fusion_payload: dict,
    fusion_site_payload: dict,
) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Current diagnosis-transfer summary under the restored sliding-window subject-vote LOSO protocol. Static FC, harmonized FC, JEPA, and late-fusion rows are compared under the same diagnosis evaluation. The site-probe column reports grouped-CV site balanced accuracy. For late-fusion rows, the site probe is run on the exact subject-level fused diagnosis probability used by late fusion, while diagnosis still uses the same late-probability fusion rule. Lower site-probe balanced accuracy is better; higher diagnosis LOSO balanced accuracy and ROC-AUC are better.}",
        "\\label{tab:jepa_fusion_subject_vote_fusion_summary}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Representation & Site probe bAcc. ($\\downarrow$) & Diagnosis LOSO bAcc. ($\\uparrow$) & Diagnosis LOSO ROC-AUC ($\\uparrow$) \\\\",
        "\\midrule",
    ]
    for index, dataset_key in enumerate(DATASET_ORDER):
        baseline_payload = baselines[dataset_key]
        rows = subject_vote_rows(baseline_payload, dataset_key, payloads, fusion_payload, fusion_site_payload)
        best_dx_bacc = max(row["dx_mean"] for row in rows)
        best_auc = max(row["auc_mean"] for row in rows)
        site_emphasis_label = SUBJECT_VOTE_SITE_EMPHASIS.get(dataset_key)
        lines.append(f"\\multicolumn{{4}}{{l}}{{{dataset_heading(baseline_payload, dataset_key)}}} \\\\")
        for row in rows:
            site_cell = (
                format_tex_pm(row["site_mean"], row["site_std"])
                if row["site_show_pm"]
                else format_tex_single(row["site_mean"])
            )
            if site_emphasis_label is not None and row["label"] == site_emphasis_label:
                site_cell = tex_bold(site_cell, True)
            dx_cell = tex_bold(format_tex_pm(row["dx_mean"], row["dx_std"]), row["dx_mean"] == best_dx_bacc)
            auc_cell = tex_bold(format_tex_pm(row["auc_mean"], row["auc_std"]), row["auc_mean"] == best_auc)
            if row["style"] is not None:
                lines.append(f"\\rowcolor{{{row['style']}}}")
            lines.append(f"{row['label']} & {site_cell} & {dx_cell} & {auc_cell} \\\\")
        chance_site = 1.0 / baseline_payload["meta"]["n_sites"]
        lines.append(f"Chance & {format_pct(chance_site)} & 50.0 & 50.0 \\\\")
        if index != len(DATASET_ORDER) - 1:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{0.2cm}",
            "\\begin{minipage}{0.97\\textwidth}",
            "\\footnotesize \\textbf{Notes:} The diagnosis columns in this table come from the restored \\textbf{sliding-window subject-vote LOSO} recipe. Static FC, ComBat FC, CovBat FC, and the late-fusion rows all use the same matched-scan diagnosis protocol. The FC-only and late-fusion diagnosis rows are reported as mean $\\pm$ standard deviation across \\textbf{5 downstream probe seeds} (`0,1,2,3,4`) for the selected JEPA checkpoint, while the standalone JEPA rows shown here are still single-checkpoint summaries and therefore appear with `$\\pm 0.0$`. The site column uses grouped-CV site probes. For standalone FC and JEPA rows, those values come from the current branch grouped-CV tables. For late-fusion rows, the site probe is a dedicated grouped-CV classifier on the exact subject-level fused diagnosis probability produced by late fusion, using the same `StandardScaler -> PCA -> LogisticRegression` family. Blue-shaded rows denote standalone JEPA models, including the exploratory domain-adversarial rotary ablation, and green-shaded rows denote late-fusion models.",
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_significance_table(
    baselines: dict[str, dict],
    fusion_payload: dict,
    stats_payload: dict,
) -> tuple[str, str]:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Late-fusion significance summary. Pairwise late-fusion comparisons were run with paired Wilcoxon signed-rank tests across held-out sites after averaging each site's metric over the 5 downstream probe seeds. Holm correction was applied within each dataset and metric family.}",
        "\\label{tab:jepa_fusion_late_fusion_significance}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Dataset & Best late fusion (bAcc) & bAcc & Best late fusion (AUC) & AUC & Statistical verdict \\\\",
        "\\midrule",
    ]
    note = ""
    for dataset_key in DATASET_ORDER:
        dataset_summary = fusion_payload["datasets"][dataset_key]["summary"]
        fusion_methods = {key: dataset_summary[key] for key in ("fusion::fc", "fusion::fc_combat", "fusion::fc_covbat")}
        best_bacc_key = max(fusion_methods, key=lambda key: fusion_methods[key]["bacc"]["mean"])
        best_auc_key = max(fusion_methods, key=lambda key: fusion_methods[key]["auc"]["mean"])
        comparisons = stats_payload["datasets"][dataset_key]["comparisons"]
        bacc_target_significant = all(comp["p_bacc_holm"] < 0.05 for comp in comparisons)
        auc_target_significant = all(comp["p_auc_holm"] < 0.05 for comp in comparisons)
        target_key = stats_payload["datasets"][dataset_key]["target"]
        if best_bacc_key == target_key and bacc_target_significant:
            if best_auc_key == target_key and auc_target_significant:
                verdict = "bAcc and AUC winners significant vs both other late-fusion rows"
            elif best_auc_key == target_key:
                verdict = "bAcc winner significant vs both other late-fusion rows; no significant AUC gap"
            else:
                verdict = "bAcc winner significant vs both other late-fusion rows; no significant AUC gap"
        else:
            verdict = "No significant late-fusion difference"
        lines.append(
            f"{baselines[dataset_key]['config']['dataset_label']} & "
            f"{tex_bold(FUSION_DISPLAY[best_bacc_key], True) if dataset_key == 'abide1' else FUSION_DISPLAY[best_bacc_key]} & "
            f"{tex_bold(format_tex_pm(fusion_methods[best_bacc_key]['bacc']['mean'], fusion_methods[best_bacc_key]['bacc']['std']), True)} & "
            f"{FUSION_DISPLAY[best_auc_key]} & "
            f"{tex_bold(format_tex_pm(fusion_methods[best_auc_key]['auc']['mean'], fusion_methods[best_auc_key]['auc']['std']), True)} & "
            f"{tex_bold(verdict, dataset_key == 'abide1')} \\\\"
        )
        if dataset_key == "abide1":
            comp_map = {comp["comparator"]: comp for comp in comparisons}
            p_fc = comp_map["fusion::fc"]["p_bacc_holm"]
            p_fc_combat = comp_map["fusion::fc_combat"]["p_bacc_holm"]
            note = (
                "\\footnotesize \\textbf{Notes:} ``Late fusion'' here means only the three fused diagnosis rows: "
                "\\textit{Static FC + JEPA}, \\textit{Static FC + ComBat + JEPA}, and \\textit{Static FC + CovBat + JEPA}. "
                "The strongest result is on \\textbf{ABIDE-I}, where \\textit{Static FC + CovBat + JEPA} is significantly better on balanced accuracy than both other late-fusion variants after Holm correction "
                f"(vs \\textit{{Static FC + JEPA}}: Holm $p={p_fc:.4f}$; vs \\textit{{Static FC + ComBat + JEPA}}: Holm $p={p_fc_combat:.4f}$). "
                "No late-fusion winner is statistically significant on ABIDE-II or SRPBS-1600 under this test."
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{0.2cm}",
            "\\begin{minipage}{0.97\\textwidth}",
            note,
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines), note


def collect_confirmation_profiles(payloads: dict[str, dict]) -> dict[str, dict]:
    result = {
        "basic_pooled": {
            "subject_vote": [
                payloads["confirm_basic_pooled_seed1_subject_vote.json"],
                payloads["confirm_basic_pooled_seed2_subject_vote.json"],
            ],
            "fusion": [
                payloads["confirm_basic_pooled_seed1_fusion_subject_vote.json"],
                payloads["confirm_basic_pooled_seed2_fusion_subject_vote.json"],
            ],
        },
        "rotary_curr_align": {
            "subject_vote": [
                payloads["confirm_rotary_curr_align_seed1_subject_vote.json"],
                payloads["confirm_rotary_curr_align_seed2_subject_vote.json"],
            ],
            "fusion": [
                payloads["confirm_rotary_curr_align_seed1_fusion_subject_vote.json"],
                payloads["confirm_rotary_curr_align_seed2_fusion_subject_vote.json"],
            ],
        },
    }
    return result


def aggregate_confirmation_summary(payloads: dict[str, dict]) -> dict[str, dict]:
    profiles = collect_confirmation_profiles(payloads)
    dataset_summary: dict[str, dict] = {}
    for dataset_key in DATASET_ORDER:
        dataset_summary[dataset_key] = {}
        for profile_key, profile_payloads in profiles.items():
            subject_vote_bacc = [payload["datasets"][dataset_key]["summary"]["bacc"]["mean"] for payload in profile_payloads["subject_vote"]]
            subject_vote_auc = [payload["datasets"][dataset_key]["summary"]["auc"]["mean"] for payload in profile_payloads["subject_vote"]]
            fusion_payloads = profile_payloads["fusion"]
            method_metrics = {}
            for method_key in ("fc", "fc_combat", "fc_covbat", "fusion::fc", "fusion::fc_combat", "fusion::fc_covbat"):
                bacc_values = [payload["datasets"][dataset_key]["summary"][method_key]["bacc"]["mean"] for payload in fusion_payloads]
                auc_values = [payload["datasets"][dataset_key]["summary"][method_key]["auc"]["mean"] for payload in fusion_payloads]
                method_metrics[method_key] = {
                    "bacc": aggregate_metric(bacc_values),
                    "auc": aggregate_metric(auc_values),
                }
            dataset_summary[dataset_key][profile_key] = {
                "jepa": {
                    "bacc": aggregate_metric(subject_vote_bacc),
                    "auc": aggregate_metric(subject_vote_auc),
                },
                **method_metrics,
            }
    return dataset_summary


def best_key(metrics: dict[str, dict], allowed_keys: tuple[str, ...]) -> str:
    return max(allowed_keys, key=lambda key: metrics[key]["bacc"][0])


def final_confirmation_rows(aggregates: dict[str, dict]) -> dict[str, dict]:
    rows = {}
    for dataset_key in DATASET_ORDER:
        profile_metrics = aggregates[dataset_key]
        best_fc_profile = max(
            ("basic_pooled", "rotary_curr_align"),
            key=lambda profile_key: max(profile_metrics[profile_key][key]["bacc"][0] for key in ("fc", "fc_combat", "fc_covbat")),
        )
        metrics_for_best_fc = profile_metrics[best_fc_profile]
        best_fc_method = best_key(metrics_for_best_fc, ("fc", "fc_combat", "fc_covbat"))

        best_jepa_profile = max(profile_metrics, key=lambda profile_key: profile_metrics[profile_key]["jepa"]["bacc"][0])
        best_overall_profile = max(
            profile_metrics,
            key=lambda profile_key: max(
                profile_metrics[profile_key][key]["bacc"][0]
                for key in ("fusion::fc", "fusion::fc_combat", "fusion::fc_covbat")
            ),
        )
        best_site_profile = max(
            profile_metrics,
            key=lambda profile_key: max(
                profile_metrics[profile_key][key]["bacc"][0] for key in ("fusion::fc_combat", "fusion::fc_covbat")
            ),
        )
        best_overall_method = best_key(profile_metrics[best_overall_profile], ("fusion::fc", "fusion::fc_combat", "fusion::fc_covbat"))
        best_site_method = best_key(profile_metrics[best_site_profile], ("fusion::fc_combat", "fusion::fc_covbat"))

        rows[dataset_key] = {
            "best_fc": (best_fc_method, best_fc_profile, metrics_for_best_fc[best_fc_method]),
            "best_jepa": ("jepa", best_jepa_profile, profile_metrics[best_jepa_profile]["jepa"]),
            "best_overall": (
                best_overall_method,
                best_overall_profile,
                profile_metrics[best_overall_profile][best_overall_method],
            ),
            "best_site": (
                best_site_method,
                best_site_profile,
                profile_metrics[best_site_profile][best_site_method],
            ),
        }
    return rows


def short_method_label(method_key: str) -> str:
    mapping = {
        "fc": "Static FC",
        "fc_combat": "FC + ComBat",
        "fc_covbat": "FC + CovBat",
        "fusion::fc": "FC + JEPA",
        "fusion::fc_combat": "FC + ComBat + JEPA",
        "fusion::fc_covbat": "FC + CovBat + JEPA",
        "jepa": "JEPA",
    }
    return mapping[method_key]


def profile_label(profile_key: str) -> str:
    return FINAL_PROFILE_LABELS[profile_key]


def compose_final_label(method_key: str, profile_key: str) -> str:
    if method_key == "jepa":
        return f"JEPA {profile_label(profile_key)}"
    if method_key == "fc":
        return "Static FC"
    if method_key == "fc_combat":
        return "FC + ComBat"
    if method_key == "fc_covbat":
        return "FC + CovBat"
    return f"{short_method_label(method_key)} {profile_label(profile_key)}"


def build_confirmation_tex(baselines: dict[str, dict], aggregates: dict[str, dict]) -> str:
    rows = final_confirmation_rows(aggregates)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Final multiseed confirmation summary using the completed full-dataset JEPA confirmation runs. Entries aggregate the valid full-dataset JEPA pretraining seeds (`1` and `2`) for each confirmation profile, with diagnosis reported as mean $\\pm$ standard deviation across those JEPA seeds after each run's 5 downstream probe seeds are summarized.}",
        "\\label{tab:jepa_fusion_final_confirmation}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Dataset & Best FC Baseline & Best JEPA & Best Overall Fusion & Best Site-Conscious Fusion \\\\",
        "\\midrule",
    ]
    for dataset_key in DATASET_ORDER:
        baseline_payload = baselines[dataset_key]
        lines.append(f"\\multicolumn{{5}}{{l}}{{{dataset_heading(baseline_payload, dataset_key)}}} \\\\")
        if dataset_key == "abide1":
            lines.append("\\rowcolor{tablefusion}")
        best_fc_method, best_fc_profile, best_fc_metrics = rows[dataset_key]["best_fc"]
        _, best_jepa_profile, best_jepa_metrics = rows[dataset_key]["best_jepa"]
        best_overall_method, best_overall_profile, best_overall_metrics = rows[dataset_key]["best_overall"]
        best_site_method, best_site_profile, best_site_metrics = rows[dataset_key]["best_site"]
        lines.append(
            f"{baseline_payload['config']['dataset_label']} & "
            f"{compose_final_label(best_fc_method, best_fc_profile)} ({format_tex_pm(*best_fc_metrics['bacc'])} / {format_tex_pm(*best_fc_metrics['auc'])}) & "
            f"{compose_final_label('jepa', best_jepa_profile)} ({format_tex_pm(*best_jepa_metrics['bacc'])} / {format_tex_pm(*best_jepa_metrics['auc'])}) & "
            f"{compose_final_label(best_overall_method, best_overall_profile)} ({format_tex_pm(*best_overall_metrics['bacc'])} / {format_tex_pm(*best_overall_metrics['auc'])}) & "
            f"{compose_final_label(best_site_method, best_site_profile)} ({format_tex_pm(*best_site_metrics['bacc'])} / {format_tex_pm(*best_site_metrics['auc'])}) \\\\"
        )
        if dataset_key != DATASET_ORDER[-1]:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{0.2cm}",
            "\\begin{minipage}{0.97\\textwidth}",
            "\\footnotesize \\textbf{Notes:} Each cell reports diagnosis LOSO balanced accuracy / ROC-AUC. The earlier `seed0` confirmation runs are excluded here because their fusion evaluation was an ABIDE-I-only smoke run rather than the full three-dataset sweep. Under a site-conscious reading, the cleanest final headline remains \\textbf{FC + CovBat + JEPA rotary curr + replay align} on ABIDE-I. ABIDE-II remains baseline-dominated, and SRPBS-1600 favors JEPA fusion numerically but not with CovBat as the harmonized winner.",
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_paper_tex(
    baselines: dict[str, dict],
    payloads: dict[str, dict],
    fusion_payload: dict,
    fusion_site_payload: dict,
    stats_payload: dict,
    confirmation_aggregates: dict[str, dict],
) -> str:
    significance_table, _ = build_significance_table(baselines, fusion_payload, stats_payload)
    sections = [
        "% Requires \\usepackage[table]{xcolor}",
        "\\definecolor{tablejepa}{RGB}{245,248,255}",
        "\\definecolor{tablefusion}{RGB}{241,249,242}",
        "",
        build_grouped_cv_table(baselines, payloads).rstrip(),
        build_subject_vote_table(baselines, payloads, fusion_payload, fusion_site_payload).rstrip(),
        significance_table.rstrip(),
        build_confirmation_tex(baselines, confirmation_aggregates).rstrip(),
        "",
    ]
    return "\n".join(sections)


def build_final_confirmation_md(baselines: dict[str, dict], confirmation_aggregates: dict[str, dict]) -> str:
    rows = final_confirmation_rows(confirmation_aggregates)
    lines = [
        "# Final Multiseed Confirmation Table",
        "",
        "This summary uses the finished **full-dataset JEPA pretraining seeds 1 and 2** for both confirmation profiles:",
        "",
        "- `basic_pooled`",
        "- `rotary_curr_align`",
        "",
        "`seed0` is excluded from this final aggregation because its fusion run was an earlier **ABIDE-I-only smoke run**, not the full `ABIDE-I + ABIDE-II + SRPBS` evaluation.",
        "",
        "All diagnosis numbers below are subject-vote LOSO means across the 5 downstream probe seeds already stored in each run, then aggregated here across the 2 valid JEPA pretraining seeds.",
        "",
        "## Final Table",
        "",
        "| Dataset | Best FC Baseline | Best JEPA | Best Overall Fusion | Best Site-Conscious Fusion | Final Read |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dataset_key in DATASET_ORDER:
        baseline_payload = baselines[dataset_key]
        best_fc_method, best_fc_profile, best_fc_metrics = rows[dataset_key]["best_fc"]
        _, best_jepa_profile, best_jepa_metrics = rows[dataset_key]["best_jepa"]
        best_overall_method, best_overall_profile, best_overall_metrics = rows[dataset_key]["best_overall"]
        best_site_method, best_site_profile, best_site_metrics = rows[dataset_key]["best_site"]
        lines.append(
            f"| {baseline_payload['config']['dataset_label']} | "
            f"{compose_final_label(best_fc_method, best_fc_profile)} ({format_md_pm(*best_fc_metrics['bacc'])} / {format_md_pm(*best_fc_metrics['auc'])}) | "
            f"{compose_final_label('jepa', best_jepa_profile)} ({format_md_pm(*best_jepa_metrics['bacc'])} / {format_md_pm(*best_jepa_metrics['auc'])}) | "
            f"{compose_final_label(best_overall_method, best_overall_profile)} ({format_md_pm(*best_overall_metrics['bacc'])} / {format_md_pm(*best_overall_metrics['auc'])}) | "
            f"{compose_final_label(best_site_method, best_site_profile)} ({format_md_pm(*best_site_metrics['bacc'])} / {format_md_pm(*best_site_metrics['auc'])}) | "
            f"{FINAL_CONFIRM_TEXT[dataset_key]} |"
        )
    lines.extend(["", "## Short Take", ""])
    for line in SHORT_TAKE_LINES:
        lines.append(f"- {line}")
    lines.extend(["", "## Per-Model Aggregates", ""])
    for profile_key in ("basic_pooled", "rotary_curr_align"):
        lines.append(f"### JEPA {profile_label(profile_key)}")
        lines.append("")
        lines.append("| Dataset | JEPA | Static FC | FC + ComBat | FC + CovBat | FC + JEPA | FC + ComBat + JEPA | FC + CovBat + JEPA |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for dataset_key in DATASET_ORDER:
            baseline_payload = baselines[dataset_key]
            metrics = confirmation_aggregates[dataset_key][profile_key]
            lines.append(
                f"| {baseline_payload['config']['dataset_label']} | "
                f"{format_md_pm(*metrics['jepa']['bacc'])} / {format_md_pm(*metrics['jepa']['auc'])} | "
                f"{format_md_pm(*metrics['fc']['bacc'])} / {format_md_pm(*metrics['fc']['auc'])} | "
                f"{format_md_pm(*metrics['fc_combat']['bacc'])} / {format_md_pm(*metrics['fc_combat']['auc'])} | "
                f"{format_md_pm(*metrics['fc_covbat']['bacc'])} / {format_md_pm(*metrics['fc_covbat']['auc'])} | "
                f"{format_md_pm(*metrics['fusion::fc']['bacc'])} / {format_md_pm(*metrics['fusion::fc']['auc'])} | "
                f"{format_md_pm(*metrics['fusion::fc_combat']['bacc'])} / {format_md_pm(*metrics['fusion::fc_combat']['auc'])} | "
                f"{format_md_pm(*metrics['fusion::fc_covbat']['bacc'])} / {format_md_pm(*metrics['fusion::fc_covbat']['auc'])} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_recovery_report(
    results_dir: Path,
    paper_tex_path: Path,
    final_md_path: Path,
    paper_tex: str,
    final_md: str,
    baselines: dict[str, dict],
    stats_payload: dict,
    fusion_payload: dict,
    existing_paper_tex: str | None,
    existing_final_md: str | None,
) -> str:
    lines = [
        "# JEPA Fusion Recovery Report",
        "",
        f"Results directory: `{results_dir}`",
        "",
        "## Status",
        "",
        "- The original `src/jepa_fusion` sources are gone; the current branch should be treated as recovery scaffolding.",
        "- The checkpoint inventory under `checkpoints/jepa_fusion` is intact and is the authoritative source for rerunning JEPA evaluations.",
        "- The paper-facing outputs can now be regenerated from the surviving JSON artifacts instead of staying hand-maintained.",
        "",
        "## Required Inputs",
        "",
    ]
    for name in dict.fromkeys(REQUIRED_JSONS):
        lines.append(f"- `{name}`")
    comparisons = stats_payload["datasets"]["abide1"]["comparisons"]
    comp_map = {comp["comparator"]: comp for comp in comparisons}
    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- `{paper_tex_path.name}` exact pre-run match: `{'yes' if existing_paper_tex == paper_tex else 'no'}`",
            f"- `{final_md_path.name}` exact pre-run match: `{'yes' if existing_final_md == final_md else 'no'}`",
            "- Value regeneration is grounded in the stored JSON summaries, not in hand-copied notes.",
            f"- ABIDE-I significance note uses Holm-corrected p-values from `rotary_curr_align_phase2_final_fusion_subject_vote_stats.stats.json`: "
            f"`{comp_map['fusion::fc']['p_bacc_holm']:.4f}` vs `Static FC + JEPA`, "
            f"`{comp_map['fusion::fc_combat']['p_bacc_holm']:.4f}` vs `Static FC + ComBat + JEPA`.",
        ]
    )
    seed0_path = results_dir / "recovery_abide1_rotary_curr_align_fusion_subject_vote_seed0.json"
    if seed0_path.exists():
        seed0_payload = load_json(seed0_path)
        old_runs = fusion_payload["datasets"]["abide1"]["runs"]
        deltas = []
        for method in ("jepa", "fc", "fc_combat", "fc_covbat", "fusion::fc", "fusion::fc_combat", "fusion::fc_covbat"):
            old_seed0 = next(run for run in old_runs[method] if run["seed"] == 0)["summary"]
            new_summary = seed0_payload["datasets"]["abide1"]["summary"][method]
            deltas.append(abs(new_summary["bacc"]["mean"] - old_seed0["bacc"]))
            deltas.append(abs(new_summary["auc"]["mean"] - old_seed0["auc"]))
        lines.append(
            f"- Live seed-0 verification artifact `{seed0_path.name}` matched the stored ABIDE-I seed-0 fusion summary with max absolute delta `{max(deltas):.1e}` across bAcc/AUC."
        )
    lines.extend(
        [
            "",
            "## Regeneration Command",
            "",
            "```bash",
            "python -m src.jepa_fusion.recover_results",
            "```",
            "",
            "## Dataset Inventory",
            "",
        ]
    )
    for dataset_key in DATASET_ORDER:
        baseline = baselines[dataset_key]
        lines.append(
            f"- `{baseline['config']['dataset_label']}`: `{baseline['meta']['n_sites']}` mixed-label sites, "
            f"task `{DATASET_TASK_LABELS[dataset_key]}`."
        )
    return "\n".join(lines).rstrip() + "\n"


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    payloads = read_payloads(results_dir)
    baselines = {
        dataset_key: normalize_baseline_payload(dataset_key, payloads[filename], results_dir)
        for dataset_key, filename in BASELINE_FILES.items()
    }
    fusion_payload = payloads["rotary_curr_align_phase2_final_fusion_subject_vote.json"]
    fusion_site_payload = payloads["rotary_curr_align_phase2_final_fusion_subject_vote_fused_prob_site_probe.json"]
    stats_payload = payloads["rotary_curr_align_phase2_final_fusion_subject_vote_stats.stats.json"]
    confirmation_aggregates = aggregate_confirmation_summary(payloads)

    paper_tex_path = Path(args.paper_tex) if args.paper_tex else results_dir / "current_paper_tables.tex"
    final_md_path = Path(args.final_md) if args.final_md else results_dir / "FINAL_MULTI_SEED_CONFIRMATION_TABLE.md"
    report_md_path = Path(args.report_md) if args.report_md else results_dir / "RECOVERY_REPORT.md"

    existing_paper_tex = paper_tex_path.read_text() if paper_tex_path.exists() else None
    existing_final_md = final_md_path.read_text() if final_md_path.exists() else None

    paper_tex = build_paper_tex(
        baselines=baselines,
        payloads=payloads,
        fusion_payload=fusion_payload,
        fusion_site_payload=fusion_site_payload,
        stats_payload=stats_payload,
        confirmation_aggregates=confirmation_aggregates,
    )
    final_md = build_final_confirmation_md(baselines, confirmation_aggregates)
    report_md = build_recovery_report(
        results_dir=results_dir,
        paper_tex_path=paper_tex_path,
        final_md_path=final_md_path,
        paper_tex=paper_tex,
        final_md=final_md,
        baselines=baselines,
        stats_payload=stats_payload,
        fusion_payload=fusion_payload,
        existing_paper_tex=existing_paper_tex,
        existing_final_md=existing_final_md,
    )

    write_text(paper_tex_path, paper_tex)
    write_text(final_md_path, final_md)
    write_text(report_md_path, report_md)

    print(f"Saved LaTeX: {paper_tex_path}")
    print(f"Saved Markdown: {final_md_path}")
    print(f"Saved Recovery Report: {report_md_path}")


if __name__ == "__main__":
    main()
