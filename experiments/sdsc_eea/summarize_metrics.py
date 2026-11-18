"""Train, validation, and full exact-match for four zero-shot runs.

Joins each model's ``results.csv`` to the SDSC-EEA dataset split and
bootstraps a 95% interval. Parse failures stored as ``[]`` are counted
separately. Prints the table used for the user-run model breakdown.
"""

from __future__ import annotations

import csv
import json
import random

from tqdm.auto import tqdm

from scaling_surgical_ai import paths

RANDOM_SEED = 42
DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
ZERO_SHOT_VLM_DIR = paths.experiment_output_dir("sdsc_eea_zero_shot_vlm")

MODELS = [
    {
        "name": "Llama-3.2-90B-Vision_temp0_0",
        "results": [
            paths.experiment_output_dir("sdsc_eea_zero_shot_llama_vision")
            / "llama-3.2-90b-vision_temp0_0"
            / "results.csv"
        ],
    },
    {
        "name": "MedGemma-3-27B-it_temp0_0",
        "results": [
            paths.experiment_output_dir("sdsc_eea_zero_shot_medgemma")
            / "medgemma-27b-it_temp0_0"
            / "results.csv"
        ],
    },
    {
        "name": "LLaVA-1.5-13B_temp0_2",
        "results": [
            paths.experiment_output_dir("sdsc_eea_zero_shot_llava")
            / "llava-v1.5-13b_temp0_2"
            / "results.csv"
        ],
    },
    {
        "name": "Qwen3-VL-235B_temp0_0",
        "results": [
            ZERO_SHOT_VLM_DIR / f"qwen3-vl-235b_temp0_0_split{i}" / "results.csv"
            for i in range(1, 5)
        ],
    },
]


def bootstrap_ci(
    correct_array,
    n_bootstrap=1000,
    confidence=0.95,
    seed=RANDOM_SEED,
):
    """Bootstrap a percentile interval for a 0/1 correctness vector.

    Args:
        correct_array: 1 for an exact match and 0 otherwise.
        n_bootstrap: Number of resamples.
        confidence: Interval mass.
        seed: Seed for the resampler.

    Returns:
        Lower and upper bounds as proportions.
    """
    if len(correct_array) == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(correct_array)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(correct_array) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1 - confidence
    lo = int(n_bootstrap * (alpha / 2))
    hi = int(n_bootstrap * (1 - alpha / 2))
    return means[lo], means[hi]


def load_split_map():
    """Map each frame path to ``train`` or ``val``.

    Returns:
        Dictionary from ``frame_path`` to the dataset split.
    """
    split_map = {}
    with DATASET_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_map[row["frame_path"]] = row["split"]
    print(f"Loaded split map: {len(split_map)} frames")
    return split_map


def count_parse_failures(results_csvs):
    """Count rows whose predicted tools are an empty list.

    Args:
        results_csvs: Result CSV paths for one model.

    Returns:
        The number of rows read and the number of empty predictions.
    """
    total = 0
    parse_fail = 0
    for rp in results_csvs:
        if not rp.exists():
            continue
        with rp.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="parse failures"):
                total += 1
                pred = row.get("predicted_tools", "")
                if pred == "[]":
                    parse_fail += 1
    return total, parse_fail


def process_model(model_info, split_map):
    """Score one model on train, validation, and all joined frames.

    Args:
        model_info: Name and result-CSV paths.
        split_map: Frame path to ``train`` or ``val``.

    Returns:
        One summary dictionary for the printed table.
    """
    name = model_info["name"]
    results_csvs = model_info["results"]

    train_em = []
    val_em = []
    all_em = []

    for rp in results_csvs:
        if not rp.exists():
            print(f"  WARNING: {rp} not found")
            continue
        with rp.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=name):
                gt_str = row.get("ground_truth_tools", "")
                pred_str = row.get("predicted_tools", "")
                split = split_map.get(row["frame_path"])

                if not gt_str or not pred_str:
                    match = 0
                else:
                    try:
                        gt = set(json.loads(gt_str))
                        pred = set(json.loads(pred_str))
                        match = 1 if gt == pred else 0
                    except (json.JSONDecodeError, TypeError):
                        match = 0

                all_em.append(match)
                if split == "train":
                    train_em.append(match)
                elif split == "val":
                    val_em.append(match)

    total, parse_fail = count_parse_failures(results_csvs)
    pf_rate = parse_fail / total * 100 if total > 0 else 0

    val_ci = bootstrap_ci(val_em)
    train_ci = bootstrap_ci(train_em)

    print(
        f"{name}: total={len(all_em)}, train={len(train_em)}, val={len(val_em)}, "
        f"parse_fail={parse_fail} ({pf_rate:.1f}%)"
    )
    print(
        f"  Train EM: {sum(train_em) / len(train_em) * 100:.2f}% "
        f"({train_ci[0] * 100:.2f}%--{train_ci[1] * 100:.2f}%)"
    )
    print(
        f"  Val EM:   {sum(val_em) / len(val_em) * 100:.2f}% "
        f"({val_ci[0] * 100:.2f}%--{val_ci[1] * 100:.2f}%)"
    )
    print(f"  All EM:   {sum(all_em) / len(all_em) * 100:.2f}%")

    return {
        "model": name,
        "total": len(all_em),
        "parse_fail": parse_fail,
        "pf_rate": f"{pf_rate:.2f}",
        "train_n": len(train_em),
        "train_em": sum(train_em),
        "train_pct": f"{sum(train_em) / len(train_em) * 100:.2f}"
        if train_em
        else "0.00",
        "val_n": len(val_em),
        "val_em": sum(val_em),
        "val_pct": f"{sum(val_em) / len(val_em) * 100:.2f}" if val_em else "0.00",
        "val_ci_lo": f"{val_ci[0] * 100:.2f}",
        "val_ci_hi": f"{val_ci[1] * 100:.2f}",
        "all_em": sum(all_em),
        "all_pct": f"{sum(all_em) / len(all_em) * 100:.2f}" if all_em else "0.00",
    }


def main() -> None:
    """Print the train, validation, and full exact-match table."""
    split_map = load_split_map()
    results = []
    for m in MODELS:
        print(f"\nProcessing {m['name']}...")
        r = process_model(m, split_map)
        results.append(r)

    print("\n\nCSV OUTPUT:")
    header = (
        "model,total,parse_fail,pf_rate,train_n,train_em,train_pct,"
        "val_n,val_em,val_pct,val_ci_lo,val_ci_hi,all_em,all_pct"
    )
    print(header)
    for r in results:
        cols = [str(r[k]) for k in header.split(",")]
        print(",".join(cols))


if __name__ == "__main__":
    main()
