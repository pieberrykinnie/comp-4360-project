import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_LINE_RE = re.compile(
    r"Train:\s+\[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?"
    r"loss\s+([0-9eE+\-.]+)\s+\(\s*([0-9eE+\-.]+)\)"
)


VAL_LINE_RE = re.compile(
    r"Validation\s*-\s*Epoch\s+(\d+):\s*mean_auc:\s*([0-9eE+\-.]+)\s*\|\s*loss:\s*([0-9eE+\-.]+)"
)


def parse_train_log(log_path):
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = TRAIN_LINE_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "epoch": int(match.group(1)),
                    "step": int(match.group(3)),
                    "loss_avg": float(match.group(6)),
                }
            )
    return rows


def parse_validation_log(log_path):
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = VAL_LINE_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "epoch": int(match.group(1)),
                    "mean_auc": float(match.group(2)),
                    "val_loss": float(match.group(3)),
                }
            )
    return rows


def average_last_loss_by_epoch(rank0_rows, rank1_rows):
    def last_by_epoch(rows):
        epoch_to_row = {}
        for row in rows:
            epoch = row["epoch"]
            if epoch not in epoch_to_row or row["step"] > epoch_to_row[epoch]["step"]:
                epoch_to_row[epoch] = row
        return {epoch: row["loss_avg"] for epoch, row in epoch_to_row.items()}

    rank0_loss = last_by_epoch(rank0_rows)
    rank1_loss = last_by_epoch(rank1_rows)

    all_epochs = sorted(set(rank0_loss.keys()) | set(rank1_loss.keys()))
    avg_loss = []

    for epoch in all_epochs:
        values = []
        if epoch in rank0_loss:
            values.append(rank0_loss[epoch])
        if epoch in rank1_loss:
            values.append(rank1_loss[epoch])
        avg_loss.append(sum(values) / len(values))

    return all_epochs, avg_loss


def average_validation_by_epoch(rank0_rows, rank1_rows):
    grouped = defaultdict(list)
    for row in rank0_rows:
        grouped[row["epoch"]].append(row)
    for row in rank1_rows:
        grouped[row["epoch"]].append(row)

    epochs = sorted(grouped.keys())
    mean_aucs = []

    for epoch in epochs:
        values = grouped[epoch]
        mean_aucs.append(sum(v["mean_auc"] for v in values) / len(values))

    return epochs, mean_aucs


def train_loss_from_csv(csv_path):
    grouped = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch = int(row["epoch"])
            step = int(row["step"])
            loss_avg = float(row["loss_avg"])
            if epoch not in grouped or step > grouped[epoch]["step"]:
                grouped[epoch] = {"step": step, "loss_avg": loss_avg}

    epochs = sorted(grouped.keys())
    losses = [grouped[epoch]["loss_avg"] for epoch in epochs]
    return epochs, losses


def val_auroc_from_csv(csv_path):
    rows = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch = int(row["epoch"])
            rows[epoch] = float(row["mean_auc"])

    epochs = sorted(rows.keys())
    aucs = [rows[epoch] for epoch in epochs]
    return epochs, aucs


def plot_lines(series_list, title, y_label, x_min, x_max, output_path):
    plt.figure(figsize=(10, 6))

    for series in series_list:
        plt.plot(
            series["epochs"],
            series["values"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=series["label"],
        )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.xlim(x_min, x_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate requested comparison plots from existing logs and CSV artifacts."
    )
    parser.add_argument(
        "--outdir",
        default="output/comparison_plots",
        help="Output directory for requested comparison plots.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pretrain_baseline_epochs, pretrain_baseline_loss = train_loss_from_csv(
        "output/simmim_pretrain/simmim_pretrain__vit_base__img224__100ep/plots/combined_extracted_metrics.csv"
    )

    pretrain_ablation_rank0 = parse_train_log(
        "output/simmim_pretrain_ablation/simmim_pretrain__vit_base__img224__100ep_ablation/log_rank0.txt"
    )
    pretrain_ablation_rank1 = parse_train_log(
        "output/simmim_pretrain_ablation/simmim_pretrain__vit_base__img224__100ep_ablation/log_rank1.txt"
    )
    pretrain_ablation_epochs, pretrain_ablation_loss = average_last_loss_by_epoch(
        pretrain_ablation_rank0,
        pretrain_ablation_rank1,
    )

    finetune_baseline_epochs, finetune_baseline_loss = train_loss_from_csv(
        "output/simmim_finetune/simmim_finetune__vit_base__img224__100ep/plots/combined_finetune_train_metrics.csv"
    )
    finetune_baseline_auroc_epochs, finetune_baseline_auroc = val_auroc_from_csv(
        "output/simmim_finetune/simmim_finetune__vit_base__img224__100ep/plots/finetune_validation_metrics.csv"
    )

    finetune_ablation_rank0_train = parse_train_log(
        "output/simmim_finetune_ablation/simmim_finetune__vit_base__img224__100ep_ablation/log_rank0.txt"
    )
    finetune_ablation_rank1_train = parse_train_log(
        "output/simmim_finetune_ablation/simmim_finetune__vit_base__img224__100ep_ablation/log_rank1.txt"
    )
    finetune_ablation_epochs, finetune_ablation_loss = average_last_loss_by_epoch(
        finetune_ablation_rank0_train,
        finetune_ablation_rank1_train,
    )

    finetune_ablation_rank0_val = parse_validation_log(
        "output/simmim_finetune_ablation/simmim_finetune__vit_base__img224__100ep_ablation/log_rank0.txt"
    )
    finetune_ablation_rank1_val = parse_validation_log(
        "output/simmim_finetune_ablation/simmim_finetune__vit_base__img224__100ep_ablation/log_rank1.txt"
    )
    finetune_ablation_auroc_epochs, finetune_ablation_auroc = average_validation_by_epoch(
        finetune_ablation_rank0_val,
        finetune_ablation_rank1_val,
    )

    only_finetune_rank0_train = parse_train_log(
        "output/only_finetune/finetune__vit_base__img224__200ep/log_rank0.txt"
    )
    only_finetune_rank1_train = parse_train_log(
        "output/only_finetune/finetune__vit_base__img224__200ep/log_rank1.txt"
    )
    only_finetune_epochs, only_finetune_loss = average_last_loss_by_epoch(
        only_finetune_rank0_train,
        only_finetune_rank1_train,
    )

    only_finetune_rank0_val = parse_validation_log(
        "output/only_finetune/finetune__vit_base__img224__200ep/log_rank0.txt"
    )
    only_finetune_rank1_val = parse_validation_log(
        "output/only_finetune/finetune__vit_base__img224__200ep/log_rank1.txt"
    )
    only_finetune_auroc_epochs, only_finetune_auroc = average_validation_by_epoch(
        only_finetune_rank0_val,
        only_finetune_rank1_val,
    )

    plot_lines(
        series_list=[
            {
                "label": "simmim_pretrain",
                "epochs": pretrain_baseline_epochs,
                "values": pretrain_baseline_loss,
            },
            {
                "label": "simmim_pretrain_ablation",
                "epochs": pretrain_ablation_epochs,
                "values": pretrain_ablation_loss,
            },
        ],
        title="Pretraining Loss vs Epoch (0-99)",
        y_label="Training Loss (loss_avg)",
        x_min=0,
        x_max=99,
        output_path=outdir / "pretrain_loss_epoch_0_99_baseline_vs_ablation.png",
    )

    plot_lines(
        series_list=[
            {
                "label": "simmim_finetune",
                "epochs": finetune_baseline_epochs,
                "values": finetune_baseline_loss,
            },
            {
                "label": "simmim_finetune_ablation",
                "epochs": finetune_ablation_epochs,
                "values": finetune_ablation_loss,
            },
            {
                "label": "only_finetune",
                "epochs": only_finetune_epochs,
                "values": only_finetune_loss,
            },
        ],
        title="Finetuning Loss vs Epoch (0-199)",
        y_label="Training Loss (loss_avg)",
        x_min=0,
        x_max=199,
        output_path=outdir / "finetune_loss_epoch_0_199_three_way.png",
    )

    plot_lines(
        series_list=[
            {
                "label": "simmim_finetune",
                "epochs": finetune_baseline_auroc_epochs,
                "values": finetune_baseline_auroc,
            },
            {
                "label": "simmim_finetune_ablation",
                "epochs": finetune_ablation_auroc_epochs,
                "values": finetune_ablation_auroc,
            },
            {
                "label": "only_finetune",
                "epochs": only_finetune_auroc_epochs,
                "values": only_finetune_auroc,
            },
        ],
        title="Finetuning Mean AUROC vs Epoch (0-199)",
        y_label="Mean AUROC",
        x_min=0,
        x_max=199,
        output_path=outdir / "finetune_auroc_epoch_0_199_three_way.png",
    )

    print("Created comparison plots:")
    print("- pretrain_loss_epoch_0_99_baseline_vs_ablation.png")
    print("- finetune_loss_epoch_0_199_three_way.png")
    print("- finetune_auroc_epoch_0_199_three_way.png")
    print(f"Output directory: {outdir}")

    print("Series lengths:")
    print(f"- pretrain baseline epochs: {len(pretrain_baseline_epochs)}")
    print(f"- pretrain ablation epochs: {len(pretrain_ablation_epochs)}")
    print(f"- finetune baseline loss epochs: {len(finetune_baseline_epochs)}")
    print(f"- finetune ablation loss epochs: {len(finetune_ablation_epochs)}")
    print(f"- only_finetune loss epochs: {len(only_finetune_epochs)}")
    print(f"- finetune baseline AUROC epochs: {len(finetune_baseline_auroc_epochs)}")
    print(f"- finetune ablation AUROC epochs: {len(finetune_ablation_auroc_epochs)}")
    print(f"- only_finetune AUROC epochs: {len(only_finetune_auroc_epochs)}")


if __name__ == "__main__":
    main()