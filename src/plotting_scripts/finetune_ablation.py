import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np



TRAIN_LINE_RE = re.compile(
    r"Train:\s+\[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?"
    r"loss\s+([0-9eE+\-\.]+)\s+\(\s*([0-9eE+\-\.]+)\)"
)


VAL_LINE_RE = re.compile(
    r"Validation\s*-\s*Epoch\s+(\d+):\s*mean_auc:\s*([0-9eE+\-\.]+)\s*\|\s*loss:\s*([0-9eE+\-\.]+)"
)


def parse_log_file(log_path):
    train_points = []
    val_points = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            train_match = TRAIN_LINE_RE.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                step = int(train_match.group(3))
                loss_avg = float(train_match.group(6))

                train_points.append({
                    "epoch": epoch,
                    "step": step,
                    "loss_avg": loss_avg,
                })
                continue

            val_match = VAL_LINE_RE.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                mean_auroc = float(val_match.group(2))
                val_loss = float(val_match.group(3))

                val_points.append({
                    "epoch": epoch,
                    "mean_auroc": mean_auroc,
                    "val_loss": val_loss,
                })

    return train_points, val_points


def combine_rank_logs(rank0_path, rank1_path):
    rank0_train, rank0_val = parse_log_file(rank0_path)
    rank1_train, rank1_val = parse_log_file(rank1_path)

    def get_last_train_loss_per_epoch(points):
        epoch_to_last = {}
        for p in points:
            epoch = p["epoch"]
            if epoch not in epoch_to_last or p["step"] > epoch_to_last[epoch]["step"]:
                epoch_to_last[epoch] = p
        return {epoch: value["loss_avg"] for epoch, value in epoch_to_last.items()}

    rank0_train_loss = get_last_train_loss_per_epoch(rank0_train)
    rank1_train_loss = get_last_train_loss_per_epoch(rank1_train)

    train_epochs = sorted(set(rank0_train_loss.keys()) | set(rank1_train_loss.keys()))
    train_losses = []

    for epoch in train_epochs:
        values = []
        if epoch in rank0_train_loss:
            values.append(rank0_train_loss[epoch])
        if epoch in rank1_train_loss:
            values.append(rank1_train_loss[epoch])
        train_losses.append(np.mean(values))

    val_groups = defaultdict(list)

    for p in rank0_val:
        val_groups[p["epoch"]].append(p)

    for p in rank1_val:
        val_groups[p["epoch"]].append(p)

    val_epochs = sorted(val_groups.keys())
    val_losses = []
    mean_aurocs = []

    for epoch in val_epochs:
        group = val_groups[epoch]
        val_losses.append(np.mean([x["val_loss"] for x in group]))
        mean_aurocs.append(np.mean([x["mean_auroc"] for x in group]))

    return {
        "train_epochs": np.array(train_epochs),
        "train_losses": np.array(train_losses),
        "val_epochs": np.array(val_epochs),
        "val_losses": np.array(val_losses),
        "mean_aurocs": np.array(mean_aurocs),
    }


def plot_train_loss(baseline, ablation, outdir):
    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline["train_epochs"],
        baseline["train_losses"],
        marker="o",
        linewidth=2,
        label="Baseline",
    )
    plt.plot(
        ablation["train_epochs"],
        ablation["train_losses"],
        marker="o",
        linewidth=2,
        label="Ablation",
    )
    plt.title("Fine-Tuning Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "finetune_train_loss_baseline_vs_ablation.png"), dpi=200)
    plt.close()


def plot_val_loss(baseline, ablation, outdir):
    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline["val_epochs"],
        baseline["val_losses"],
        marker="o",
        linewidth=2,
        label="Baseline",
    )
    plt.plot(
        ablation["val_epochs"],
        ablation["val_losses"],
        marker="o",
        linewidth=2,
        label="Ablation",
    )
    plt.title("Fine-Tuning Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "finetune_val_loss_baseline_vs_ablation.png"), dpi=200)
    plt.close()


def plot_mean_auroc(baseline, ablation, outdir):
    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline["val_epochs"],
        baseline["mean_aurocs"],
        marker="o",
        linewidth=2,
        label="Baseline",
    )
    plt.plot(
        ablation["val_epochs"],
        ablation["mean_aurocs"],
        marker="o",
        linewidth=2,
        label="Ablation",
    )
    plt.title("Fine-Tuning Mean AUROC vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean AUROC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "finetune_mean_auroc_baseline_vs_ablation.png"), dpi=200)
    plt.close()


def plot_best_auroc_bar(baseline, ablation, outdir):
    baseline_best = float(np.max(baseline["mean_aurocs"]))
    ablation_best = float(np.max(ablation["mean_aurocs"]))

    labels = ["Baseline", "Ablation"]
    values = [baseline_best, ablation_best]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.title("Best Mean AUROC")
    plt.ylabel("Mean AUROC")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "finetune_best_mean_auroc_bar.png"), dpi=200)
    plt.close()


def save_summary_file(baseline, ablation, outdir):
    summary_path = os.path.join(outdir, "finetune_ablation_summary.txt")

    baseline_best_auroc = float(np.max(baseline["mean_aurocs"]))
    ablation_best_auroc = float(np.max(ablation["mean_aurocs"]))

    baseline_best_val_loss = float(np.min(baseline["val_losses"]))
    ablation_best_val_loss = float(np.min(ablation["val_losses"]))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Fine-Tuning Baseline vs Ablation Summary\n")
        f.write("--------------------------------------\n")
        f.write(f"Baseline best mean AUROC: {baseline_best_auroc:.4f}\n")
        f.write(f"Ablation best mean AUROC: {ablation_best_auroc:.4f}\n")
        f.write(f"Baseline best validation loss: {baseline_best_val_loss:.4f}\n")
        f.write(f"Ablation best validation loss: {ablation_best_val_loss:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot fine-tuning baseline vs ablation results.")
    parser.add_argument("--baseline-rank0", required=True, help="Path to baseline log_rank0.txt")
    parser.add_argument("--baseline-rank1", required=True, help="Path to baseline log_rank1.txt")
    parser.add_argument("--ablation-rank0", required=True, help="Path to ablation log_rank0.txt")
    parser.add_argument("--ablation-rank1", required=True, help="Path to ablation log_rank1.txt")
    parser.add_argument("--outdir", default="finetune_ablation_plots", help="Folder to save plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    baseline = combine_rank_logs(args.baseline_rank0, args.baseline_rank1)
    ablation = combine_rank_logs(args.ablation_rank0, args.ablation_rank1)

    plot_train_loss(baseline, ablation, args.outdir)
    plot_val_loss(baseline, ablation, args.outdir)
    plot_mean_auroc(baseline, ablation, args.outdir)
    plot_best_auroc_bar(baseline, ablation, args.outdir)
    save_summary_file(baseline, ablation, args.outdir)

    print(f"Plots saved in: {args.outdir}")


if __name__ == "__main__":
    main()