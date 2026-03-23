import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_summaries(results_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(results_dir.glob("*_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_path"] = str(path)
            rows.append(data)
    return rows


def build_table(rows: List[Dict]) -> List[Dict]:
    table = []
    baseline_rows = [r for r in rows if int(r.get("world_size", 0)) == 1]
    baseline_tps = None
    if baseline_rows:
        baseline_tps = max(float(r["throughput"]["tokens_per_sec_avg"]) for r in baseline_rows)

    for r in rows:
        world_size = int(r.get("world_size", 1))
        tps = float(r["throughput"]["tokens_per_sec_avg"])
        step_time = float(r["timing"]["avg_step_time_sec"])
        max_mem = float(r["memory"]["max_allocated_bytes"])
        cv = float(r["expert_imbalance"]["cv_avg"])
        eff = None
        if baseline_tps and world_size > 0:
            eff = tps / (world_size * baseline_tps)
        table.append(
            {
                "run_name": r.get("run_name", "unknown"),
                "world_size": world_size,
                "tokens_per_sec_avg": tps,
                "max_mem_allocated_bytes": max_mem,
                "imbalance_cv_avg": cv,
                "avg_step_time_sec": step_time,
                "scaling_efficiency": eff,
            }
        )
    return sorted(table, key=lambda x: (x["world_size"], x["run_name"]))


def print_table(table: List[Dict]) -> None:
    if not table:
        print("No summary files found.")
        return
    headers = [
        "run_name",
        "world_size",
        "tokens_per_sec_avg",
        "max_mem_allocated_bytes",
        "imbalance_cv_avg",
        "avg_step_time_sec",
        "scaling_efficiency",
    ]
    print(" | ".join(headers))
    print("-" * 120)
    for row in table:
        eff = row["scaling_efficiency"]
        eff_str = f"{eff:.4f}" if eff is not None else "n/a"
        print(
            f"{row['run_name']} | {row['world_size']} | {row['tokens_per_sec_avg']:.2f} | "
            f"{row['max_mem_allocated_bytes']:.0f} | {row['imbalance_cv_avg']:.4f} | "
            f"{row['avg_step_time_sec']:.4f} | {eff_str}"
        )


def save_csv(table: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run_name",
        "world_size",
        "tokens_per_sec_avg",
        "max_mem_allocated_bytes",
        "imbalance_cv_avg",
        "avg_step_time_sec",
        "scaling_efficiency",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_csv", type=str, default="results/phase1_table.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = load_summaries(results_dir)
    table = build_table(rows)
    print_table(table)
    save_csv(table, Path(args.out_csv))
    print(f"Saved table to {args.out_csv}")


if __name__ == "__main__":
    main()

