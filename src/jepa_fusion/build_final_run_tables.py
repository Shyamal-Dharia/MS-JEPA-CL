from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import sys


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Build LaTeX tables from a tagged final paper run.")
    parser.add_argument("--run-tag", default="final_run_v1")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--seeds", default="0,8,16,32,128")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_load_json(path: Path):
    if not path.exists():
        return None
    return load_json(path)


def fmt_pct(mean_val: float, std_val: float) -> str:
    return f"${100.0 * mean_val:.1f} \\pm {100.0 * std_val:.1f}$"


def fmt_val(value: float) -> str:
    return f"${value:.1f}$"


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def write_text(path: Path, text: str):
    path.write_text(text)
    print(f"Saved {path}")


def aggregate_jepa_metric(results_dir: Path, run_names: list[str], dataset_key: str):
    site_bacc_vals = []
    site_auc_vals = []
    dx_bacc_vals = []
    dx_auc_vals = []
    n_sites = None
    n_subjects = None

    for run_name in run_names:
        groupedcv = load_json(results_dir / f"{run_name}_groupedcv.json")
        subject_vote = load_json(results_dir / f"{run_name}_subject_vote.json")

        g_dataset = groupedcv["datasets"][dataset_key]
        s_dataset = subject_vote["datasets"][dataset_key]

        site_bacc_vals.append(g_dataset["summary"]["site"]["balanced_accuracy"]["mean"])
        site_auc_vals.append(g_dataset["summary"]["site"]["roc_auc"]["mean"])
        dx_bacc_vals.append(s_dataset["summary"]["bacc"]["mean"])
        dx_auc_vals.append(s_dataset["summary"]["auc"]["mean"])

        if n_sites is None:
            n_sites = int(g_dataset["meta"]["n_sites"])
        if n_subjects is None:
            n_subjects = int(s_dataset["summary"]["n_subjects"])

    return {
        "site_bacc": mean_std(site_bacc_vals),
        "site_auc": mean_std(site_auc_vals),
        "dx_bacc": mean_std(dx_bacc_vals),
        "dx_auc": mean_std(dx_auc_vals),
        "n_sites": n_sites,
        "n_subjects": n_subjects,
    }


def aggregate_jepa_overall(results_dir: Path, run_names: list[str], dataset_keys: list[str]):
    site_bacc_vals = []
    site_auc_vals = []
    dx_bacc_vals = []
    dx_auc_vals = []

    for run_name in run_names:
        groupedcv = load_json(results_dir / f"{run_name}_groupedcv.json")
        subject_vote = load_json(results_dir / f"{run_name}_subject_vote.json")

        site_bacc_vals.append(
            mean(
                groupedcv["datasets"][dataset_key]["summary"]["site"]["balanced_accuracy"]["mean"]
                for dataset_key in dataset_keys
            )
        )
        site_auc_vals.append(
            mean(
                groupedcv["datasets"][dataset_key]["summary"]["site"]["roc_auc"]["mean"]
                for dataset_key in dataset_keys
            )
        )
        dx_bacc_vals.append(
            mean(
                subject_vote["datasets"][dataset_key]["summary"]["bacc"]["mean"]
                for dataset_key in dataset_keys
            )
        )
        dx_auc_vals.append(
            mean(
                subject_vote["datasets"][dataset_key]["summary"]["auc"]["mean"]
                for dataset_key in dataset_keys
            )
        )

    return {
        "site_bacc": mean_std(site_bacc_vals),
        "site_auc": mean_std(site_auc_vals),
        "dx_bacc": mean_std(dx_bacc_vals),
        "dx_auc": mean_std(dx_auc_vals),
    }


def build_baselines_table(results_dir: Path, run_tag: str) -> str | None:
    abide_path = results_dir / f"{run_tag}_abide_hc_asd_mixedsites_baseline_table.json"
    srpbs_path = results_dir / f"{run_tag}_srpbs_hc_disorder_mixedsites_baseline_table.json"
    if not abide_path.exists() or not srpbs_path.exists():
        return None

    abide = load_json(abide_path)
    srpbs = load_json(srpbs_path)

    method_map = [
        ("raw", "ROI Time Series"),
        ("fc", "Static FC"),
        ("fc_combat", "Static FC + ComBat"),
        ("fc_covbat", "Static FC + CovBat"),
    ]

    def section(payload: dict, title: str) -> list[str]:
        lines = [rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\[2pt]"]
        for key, label in method_map:
            lines.append(
                f"{label:<22} & "
                f"{fmt_pct(payload['summary']['site'][key]['balanced_accuracy']['mean'], payload['summary']['site'][key]['balanced_accuracy']['std'])} & "
                f"{fmt_pct(payload['summary']['site'][key]['roc_auc']['mean'], payload['summary']['site'][key]['roc_auc']['std'])} & "
                f"{fmt_pct(payload['summary']['diagnosis'][key]['balanced_accuracy']['mean'], payload['summary']['diagnosis'][key]['balanced_accuracy']['std'])} & "
                f"{fmt_pct(payload['summary']['diagnosis'][key]['roc_auc']['mean'], payload['summary']['diagnosis'][key]['roc_auc']['std'])} \\\\"
            )
        chance = 100.0 / float(payload["meta"]["n_sites"])
        lines.append(r"\midrule")
        lines.append(
            f"Chance               & {fmt_val(chance)} & {fmt_val(50.0)} & {fmt_val(50.0)} & {fmt_val(50.0)} \\\\"
        )
        return lines

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Baseline results on ABIDE-I/II (HC vs.\ ASD) and SRPBS-1600 (HC vs.\ all disorders).",
        r"Site-probe balanced accuracy and ROC-AUC are computed with grouped subject-level cross-validation.",
        r"Diagnosis balanced accuracy and ROC-AUC are computed with leave-one-site-out evaluation.",
        r"Values are mean $\pm$ s.d.\ across 5 seeds. Chance-level site balanced accuracy equals $1/K$, where $K$ is the number of sites.}",
        r"\label{tab:baselines}",
        r"\small",
        r"\begin{tabular}{l rr rr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Site Probe}} & \multicolumn{2}{c}{\textbf{Diagnosis}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"\textbf{Method} & bAcc (\%) & ROC-AUC (\%) & bAcc (\%) & ROC-AUC (\%) \\",
        r"\midrule",
        *section(abide, r"ABIDE-I + ABIDE-II \quad (HC vs.\ ASD)"),
        r"\midrule",
        *section(srpbs, r"SRPBS-1600 \quad (HC vs.\ all disorders)"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_pilot_table(results_dir: Path, run_tag: str, seeds: list[int]) -> str | None:
    model_map = [
        ("temporal_basic", "Temporal: Basic"),
        ("st_roi_patch", "ST ROI Patch"),
        ("temporal_conv_net", "Temporal: ConvNet"),
        ("temporal_rotary_conv_net", "Temporal: Rotary ConvNet"),
    ]
    dataset_map = [("abide", "ABIDE-I/II"), ("srpbs", "SRPBS-1600")]

    rows = []
    for model_key, label in model_map:
        run_names = [f"{run_tag}_pilot_{model_key}_pooled_seed{seed}" for seed in seeds]
        if not all((results_dir / f"{run_name}_groupedcv.json").exists() for run_name in run_names):
            return None
        for dataset_key, dataset_label in dataset_map:
            agg = aggregate_jepa_metric(results_dir, run_names, dataset_key)
            rows.append((label, dataset_label, agg))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-dataset comparison of JEPA masking strategies under pooled training (60 epochs on all data).",
        r"Chance-level site balanced accuracy is $2.7\%$ for ABIDE-I/II and $9.1\%$ for SRPBS-1600.",
        r"Chance-level ROC-AUC and diagnosis balanced accuracy are $50.0\%$.",
        r"Values are mean $\pm$ s.d.\ across 5 pretraining seeds.}",
        r"\label{tab:jepa-pilot}",
        r"\small",
        r"\begin{tabular}{ll rr rr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Site Probe}} & \multicolumn{2}{c}{\textbf{Diagnosis}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"\textbf{Masking Strategy} & \textbf{Dataset} & bAcc (\%) & ROC-AUC (\%) & bAcc (\%) & ROC-AUC (\%) \\",
        r"\midrule",
    ]

    for idx, (label, dataset_label, agg) in enumerate(rows):
        if dataset_label == "ABIDE-I/II":
            lines.append(rf"\multirow{{2}}{{*}}{{{label}}}")
            lines.append(
                f"  & {dataset_label} & {fmt_pct(*agg['site_bacc'])} & {fmt_pct(*agg['site_auc'])} & {fmt_pct(*agg['dx_bacc'])} & {fmt_pct(*agg['dx_auc'])} \\\\"
            )
        else:
            lines.append(
                f"  & {dataset_label} & {fmt_pct(*agg['site_bacc'])} & {fmt_pct(*agg['site_auc'])} & {fmt_pct(*agg['dx_bacc'])} & {fmt_pct(*agg['dx_auc'])} \\\\"
            )
            if idx < len(rows) - 1:
                lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_curriculum_table(results_dir: Path, run_tag: str, seeds: list[int]) -> str | None:
    pooled_runs = [f"{run_tag}_pilot_temporal_rotary_conv_net_pooled_seed{seed}" for seed in seeds]
    curr_runs = [f"{run_tag}_rotary_curr_replay_seed{seed}" for seed in seeds]
    if not all((results_dir / f"{run_name}_groupedcv.json").exists() for run_name in pooled_runs + curr_runs):
        return None

    def dataset_section(dataset_key: str, title: str, chance: float) -> list[str]:
        pooled = aggregate_jepa_metric(results_dir, pooled_runs, dataset_key)
        curr = aggregate_jepa_metric(results_dir, curr_runs, dataset_key)
        return [
            rf"\multicolumn{{5}}{{l}}{{\textit{{{title} \quad [chance site bAcc = {chance:.1f}\%]}}}} \\[2pt]",
            f"Pooled & {fmt_pct(*pooled['site_bacc'])} & {fmt_pct(*pooled['site_auc'])} & {fmt_pct(*pooled['dx_bacc'])} & {fmt_pct(*pooled['dx_auc'])} \\\\",
            f"Curriculum + Replay & {fmt_pct(*curr['site_bacc'])} & {fmt_pct(*curr['site_auc'])} & {fmt_pct(*curr['dx_bacc'])} & {fmt_pct(*curr['dx_auc'])} \\\\",
        ]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Effect of curriculum training on the Rotary ConvNet JEPA variant for each dataset.",
        r"The curriculum configuration uses replay alignment with weight $0.1$ in both phases.",
        r"Site-probe balanced accuracy and ROC-AUC are computed with grouped subject-level cross-validation.",
        r"Diagnosis balanced accuracy and ROC-AUC are computed with leave-one-site-out evaluation and subject-level majority vote.",
        r"Values are mean $\pm$ s.d.\ across 5 pretraining seeds.}",
        r"\label{tab:curriculum}",
        r"\small",
        r"\begin{tabular}{l rr rr}",
        r"\toprule",
        r"\textbf{Training Schedule} & \multicolumn{2}{c}{\textbf{Site Probe}} & \multicolumn{2}{c}{\textbf{Diagnosis}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"& bAcc (\%) & ROC-AUC (\%) & bAcc (\%) & ROC-AUC (\%) \\",
        r"\midrule",
        *dataset_section("abide", "ABIDE-I + ABIDE-II", 2.7),
        r"\midrule",
        *dataset_section("srpbs", "SRPBS-1600", 9.1),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_aux_table(results_dir: Path, run_tag: str, seeds: list[int]) -> str | None:
    pooled_baseline_runs = [f"{run_tag}_pilot_temporal_rotary_conv_net_pooled_seed{seed}" for seed in seeds]
    curr_baseline_runs = [f"{run_tag}_rotary_curr_replay_seed{seed}" for seed in seeds]
    pooled_adv_runs = [f"{run_tag}_rotary_pooled_adv_seed{seed}" for seed in seeds]
    pooled_nce_runs = [f"{run_tag}_rotary_pooled_nce_seed{seed}" for seed in seeds]
    pooled_both_runs = [f"{run_tag}_rotary_pooled_both_seed{seed}" for seed in seeds]
    curr_adv_runs = [f"{run_tag}_rotary_curr_replay_adv_seed{seed}" for seed in seeds]
    curr_nce_runs = [f"{run_tag}_rotary_curr_replay_nce_seed{seed}" for seed in seeds]
    curr_both_runs = [f"{run_tag}_rotary_curr_replay_both_seed{seed}" for seed in seeds]

    needed = (
        pooled_baseline_runs
        + curr_baseline_runs
        + pooled_adv_runs
        + pooled_nce_runs
        + pooled_both_runs
        + curr_adv_runs
        + curr_nce_runs
        + curr_both_runs
    )
    if not all((results_dir / f"{run_name}_groupedcv.json").exists() for run_name in needed):
        return None

    configs = [
        ("Pooled", "Baseline", pooled_baseline_runs),
        ("Pooled", "Adversarial", pooled_adv_runs),
        ("Pooled", "InfoNCE", pooled_nce_runs),
        ("Pooled", "Adv + InfoNCE", pooled_both_runs),
        ("Curriculum + Replay", "Baseline", curr_baseline_runs),
        ("Curriculum + Replay", "Adversarial", curr_adv_runs),
        ("Curriculum + Replay", "InfoNCE", curr_nce_runs),
        ("Curriculum + Replay", "Adv + InfoNCE", curr_both_runs),
    ]

    def dataset_section(dataset_key: str, title: str, chance: float) -> list[str]:
        out = [rf"\multicolumn{{5}}{{l}}{{\textit{{{title} \quad [chance site bAcc = {chance:.1f}\%]}}}} \\[2pt]"]
        for schedule, loss_name, runs in configs:
            agg = aggregate_jepa_metric(results_dir, runs, dataset_key)
            out.append(
                f"{schedule:<18} & {loss_name:<15} & {fmt_pct(*agg['site_bacc'])} & {fmt_pct(*agg['dx_bacc'])} & {fmt_pct(*agg['dx_auc'])} \\\\"
            )
        return out

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Auxiliary-loss ablation for the Rotary ConvNet variant.",
        r"Curriculum rows use replay alignment with weight $0.1$ in both phases; InfoNCE and adversarial losses also use weight $0.1$ when enabled.",
        r"Site-probe balanced accuracy is computed with grouped subject-level cross-validation.",
        r"Diagnosis balanced accuracy and ROC-AUC are computed with leave-one-site-out evaluation and subject-level majority vote.",
        r"Values are mean $\pm$ s.d.\ across 5 pretraining seeds.}",
        r"\label{tab:aux-loss-ablation}",
        r"\small",
        r"\begin{tabular}{ll r rr}",
        r"\toprule",
        r"& & \textbf{Site Probe} & \multicolumn{2}{c}{\textbf{Diagnosis}} \\",
        r"\cmidrule(lr){3-3} \cmidrule(lr){4-5}",
        r"\textbf{Training Schedule} & \textbf{Auxiliary Loss} & bAcc (\%) $\downarrow$ & bAcc (\%) $\uparrow$ & ROC-AUC (\%) $\uparrow$ \\",
        r"\midrule",
        *dataset_section("abide", r"ABIDE-I + ABIDE-II", 100.0 / 37.0),
        r"\midrule",
        *dataset_section("srpbs", r"SRPBS-1600", 100.0 / 11.0),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_transfer_table(results_dir: Path, run_tag: str, seeds: list[int]) -> str | None:
    rows = [
        ("SRPBS + ABIDE-I", "ABIDE-II (unseen)", "abide2", [f"{run_tag}_transfer_srpbs_abide1_to_abide2_rotary_curr_replay_nce_seed{seed}" for seed in seeds]),
        ("SRPBS + ABIDE-II", "ABIDE-I (unseen)", "abide1", [f"{run_tag}_transfer_srpbs_abide2_to_abide1_rotary_curr_replay_nce_seed{seed}" for seed in seeds]),
    ]
    if not all((results_dir / f"{run_name}_groupedcv.json").exists() for _, _, _, run_names in rows for run_name in run_names):
        return None

    body = []
    for train_label, eval_label, dataset_key, run_names in rows:
        agg = aggregate_jepa_metric(results_dir, run_names, dataset_key)
        chance = 100.0 / float(agg["n_sites"])
        body.append(
            f"{train_label} & {eval_label} & {agg['n_sites']} & {chance:.1f} & "
            f"{fmt_pct(*agg['site_bacc'])} & {fmt_pct(*agg['dx_bacc'])} & {fmt_pct(*agg['dx_auc'])} \\\\"
        )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Zero-shot cross-dataset transfer ($K{=}0$).",
        r"The JEPA model uses a rotary conv-net encoder with curriculum training, replay alignment ($0.1$), and InfoNCE ($0.1$), with replay active in both phases.",
        r"It is pretrained on SRPBS-1600 plus one ABIDE dataset and evaluated on \emph{unseen} sites of the held-out ABIDE dataset, with overlapping centres excluded.",
        r"Site-probe balanced accuracy is computed with grouped subject-level cross-validation; chance level is $1/N_{\text{sites}}$.",
        r"Diagnosis balanced accuracy and ROC-AUC are computed with leave-one-site-out evaluation and subject-level majority vote.",
        r"Values are mean $\pm$ s.d.\ across 5 pretraining seeds.}",
        r"\label{tab:transfer}",
        r"\small",
        r"\begin{tabular}{ll c c r rr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Evaluation Set}} & \textbf{Site Probe} & \multicolumn{2}{c}{\textbf{Diagnosis}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-5} \cmidrule(lr){6-7}",
        r"\textbf{Train Data} & \textbf{Eval Data} & $N_{\text{sites}}$ & Chance (\%) & bAcc (\%) $\downarrow$ & bAcc (\%) $\uparrow$ & ROC-AUC (\%) $\uparrow$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_lowk_table(results_dir: Path, run_tag: str) -> str | None:
    abide_lowk = maybe_load_json(results_dir / f"{run_tag}_abide_lowk.json")
    srpbs_lowk = maybe_load_json(results_dir / f"{run_tag}_srpbs_lowk.json")
    abide_base = maybe_load_json(results_dir / f"{run_tag}_abide_hc_asd_mixedsites_baseline_table.json")
    srpbs_base = maybe_load_json(results_dir / f"{run_tag}_srpbs_hc_disorder_mixedsites_baseline_table.json")
    if not all([abide_lowk, srpbs_lowk, abide_base, srpbs_base]):
        return None

    def lowk_rows(lowk_payload: dict, baseline_payload: dict, title: str, chance: float) -> list[str]:
        rows = [rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}}}} \\"]
        k_values = [int(k) for k in lowk_payload["config"]["k_values"]]
        first_k = str(k_values[0])
        fc = lowk_payload["results"][first_k]["fc"]
        rows.append(r"\midrule")
        rows.append(
            f"--- & Static FC         & {fmt_pct(fc['site']['balanced_accuracy']['mean'], fc['site']['balanced_accuracy']['std'])} & "
            f"{fmt_pct(fc['site']['roc_auc']['mean'], fc['site']['roc_auc']['std'])} & "
            f"{fmt_pct(fc['diagnosis']['balanced_accuracy']['mean'], fc['diagnosis']['balanced_accuracy']['std'])} & "
            f"{fmt_pct(fc['diagnosis']['roc_auc']['mean'], fc['diagnosis']['roc_auc']['std'])} \\\\"
        )
        for k in k_values:
            key = str(k)
            combat = lowk_payload["results"][key]["fc_combat"]
            covbat = lowk_payload["results"][key]["fc_covbat"]
            rows.append(
                f"{k:<3} & FC + ComBat       & {fmt_pct(combat['site']['balanced_accuracy']['mean'], combat['site']['balanced_accuracy']['std'])} & "
                f"{fmt_pct(combat['site']['roc_auc']['mean'], combat['site']['roc_auc']['std'])} & "
                f"{fmt_pct(combat['diagnosis']['balanced_accuracy']['mean'], combat['diagnosis']['balanced_accuracy']['std'])} & "
                f"{fmt_pct(combat['diagnosis']['roc_auc']['mean'], combat['diagnosis']['roc_auc']['std'])} \\\\"
            )
            rows.append(
                f"    & FC + CovBat       & {fmt_pct(covbat['site']['balanced_accuracy']['mean'], covbat['site']['balanced_accuracy']['std'])} & "
                f"{fmt_pct(covbat['site']['roc_auc']['mean'], covbat['site']['roc_auc']['std'])} & "
                f"{fmt_pct(covbat['diagnosis']['balanced_accuracy']['mean'], covbat['diagnosis']['balanced_accuracy']['std'])} & "
                f"{fmt_pct(covbat['diagnosis']['roc_auc']['mean'], covbat['diagnosis']['roc_auc']['std'])} \\\\"
            )
        rows.append(
            f"All & FC + ComBat       & {fmt_pct(baseline_payload['summary']['site']['fc_combat']['balanced_accuracy']['mean'], baseline_payload['summary']['site']['fc_combat']['balanced_accuracy']['std'])} & "
            f"{fmt_pct(baseline_payload['summary']['site']['fc_combat']['roc_auc']['mean'], baseline_payload['summary']['site']['fc_combat']['roc_auc']['std'])} & "
            f"{fmt_pct(baseline_payload['summary']['diagnosis']['fc_combat']['balanced_accuracy']['mean'], baseline_payload['summary']['diagnosis']['fc_combat']['balanced_accuracy']['std'])} & "
            f"{fmt_pct(baseline_payload['summary']['diagnosis']['fc_combat']['roc_auc']['mean'], baseline_payload['summary']['diagnosis']['fc_combat']['roc_auc']['std'])} \\\\"
        )
        rows.append(
            f"    & FC + CovBat       & {fmt_pct(baseline_payload['summary']['site']['fc_covbat']['balanced_accuracy']['mean'], baseline_payload['summary']['site']['fc_covbat']['balanced_accuracy']['std'])} & "
            f"{fmt_pct(baseline_payload['summary']['site']['fc_covbat']['roc_auc']['mean'], baseline_payload['summary']['site']['fc_covbat']['roc_auc']['std'])} & "
            f"{fmt_pct(baseline_payload['summary']['diagnosis']['fc_covbat']['balanced_accuracy']['mean'], baseline_payload['summary']['diagnosis']['fc_covbat']['balanced_accuracy']['std'])} & "
            f"{fmt_pct(baseline_payload['summary']['diagnosis']['fc_covbat']['roc_auc']['mean'], baseline_payload['summary']['diagnosis']['fc_covbat']['roc_auc']['std'])} \\\\"
        )
        rows.append(r"\midrule")
        rows.append(
            f"Chance &                & {fmt_val(chance)} & {fmt_val(50.0)} & {fmt_val(50.0)} & {fmt_val(50.0)} \\\\"
        )
        return rows

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Appendix: low-$K$ calibration stress test.",
        r"ComBat and CovBat are fitted using $K$ randomly sampled subjects per site and then applied to all data.",
        r"Site-probe metrics are computed with grouped subject-level cross-validation.",
        r"Diagnosis metrics are computed with leave-one-site-out evaluation.",
        r"Values are mean $\pm$ s.d.\ across 5 seeds.}",
        r"\label{tab:lowk-appendix}",
        r"\scriptsize",
        r"\begin{tabular}{cl rr rr}",
        r"\toprule",
        r"$K$ & \textbf{Method} & \multicolumn{2}{c}{\textbf{Site Probe}} & \multicolumn{2}{c}{\textbf{Diagnosis}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"& & bAcc (\%) & ROC-AUC (\%) & bAcc (\%) & ROC-AUC (\%) \\",
        r"\midrule",
        *lowk_rows(srpbs_lowk, srpbs_base, r"SRPBS-1600 \quad (HC vs.\ all disorders)", 100.0 / 11.0),
        r"\midrule",
        *lowk_rows(abide_lowk, abide_base, r"ABIDE-I + ABIDE-II \quad (HC vs.\ ASD)", 100.0 / 37.0),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    results_dir = Path(args.results_dir) if args.results_dir else ROOT / "results" / "reports" / "tables" / "jepa_fusion" / args.run_tag
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        f"{args.run_tag}_baselines.tex": build_baselines_table(results_dir, args.run_tag),
        f"{args.run_tag}_jepa_pilot.tex": build_pilot_table(results_dir, args.run_tag, seeds),
        f"{args.run_tag}_curriculum.tex": build_curriculum_table(results_dir, args.run_tag, seeds),
        f"{args.run_tag}_aux_loss_ablation.tex": build_aux_table(results_dir, args.run_tag, seeds),
        f"{args.run_tag}_transfer.tex": build_transfer_table(results_dir, args.run_tag, seeds),
        f"{args.run_tag}_lowk_appendix.tex": build_lowk_table(results_dir, args.run_tag),
    }

    for filename, content in outputs.items():
        if content is None:
            print(f"Skipping {filename}: required inputs are missing")
            continue
        write_text(results_dir / filename, content)


if __name__ == "__main__":
    main()
