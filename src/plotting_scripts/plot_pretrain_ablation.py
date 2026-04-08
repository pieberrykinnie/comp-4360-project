import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


TRAIN_RE = re.compile(
    r"Train:\s+\[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?"
    r"lr([0-9eE+\-\.]+).*?"
    r"loss\s+([0-9eE+\-\.]+)\s+\(\s*([0-9eE+\-\.]+)\)\s+"
    r"grad_norm\s+([0-9eE+\-\.]+)\s+\(\s*([0-9eE+\-\.]+)\)"
)

MASK_RATIO_RE = re.compile(r"MASK_RATIO:\s*([0-9.]+)")


def parse_log(log_path):
    points = []
    mask_ratio = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if mask_ratio is None:
                mask_match = MASK_RATIO_RE.search(line)
                if mask_match:
                    mask_ratio = float(mask_match.group(1))

            match = TRAIN_RE.search(line)
            if not match:
                continue

            epoch = int(match.group(1))
            step = int(match.group(3))
            steps_per_epoch = int(match.group(4))
            lr = float(match.group(5))
            loss_avg = float(match.group(7))
            grad_current = float(match.group(8))

            global_step = epoch * steps_per_epoch + step

            points.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "steps_per_epoch": steps_per_epoch,
                    "global_step": global_step,
                    "lr": lr,
                    "loss_avg": loss_avg,
                    "grad_current": grad_current,
                }
            )

    return points, mask_ratio


def combine_rank_logs(rank0_path, rank1_path):
    rank0_points, mask0 = parse_log(rank0_path)
    rank1_points, mask1 = parse_log(rank1_path)

    all_points = rank0_points + rank1_points

    
    def last_loss_for_each_epoch(points):
        epoch_to_last = {}
        for p in points:
            epoch = p["epoch"]
            if epoch not in epoch_to_last or p["step"] > epoch_to_last[epoch]["step"]:
                epoch_to_last[epoch] = p
        return {epoch: value["loss_avg"] for epoch, value in epoch_to_last.items()}

    rank0_epoch_loss = last_loss_for_each_epoch(rank0_points)
    rank1_epoch_loss = last_loss_for_each_epoch(rank1_points)

    epochs = sorted(set(rank0_epoch_loss.keys()) | set(rank1_epoch_loss.keys()))
    epoch_losses = []

    for epoch in epochs:
        values = []
        if epoch in rank0_epoch_loss:
            values.append(rank0_epoch_loss[epoch])
        if epoch in rank1_epoch_loss:
            values.append(rank1_epoch_loss[epoch])
        epoch_losses.append(np.mean(values))


    step_groups = defaultdict(list)
    for p in all_points:
        step_groups[p["global_step"]].append(p["grad_current"])

    global_steps = sorted(step_groups.keys())
    grad_norms = [np.mean(step_groups[s]) for s in global_steps]

    return {
        "mask_ratio": mask0 if mask0 is not None else mask1,
        "epochs": np.array(epochs),
        "epoch_losses": np.array(epoch_losses),
        "global_steps": np.array(global_steps),
        "grad_norms": np.array(grad_norms),
    }


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def moving_average_x(x, window):
    if window <= 1 or len(x) < window:
        return x
    return x[window - 1:]


def plot_loss_vs_epoch(baseline, ablation, outdir):
    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline["epochs"],
        baseline["epoch_losses"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=f"Baseline (mask ratio={baseline['mask_ratio']})",
    )
    plt.plot(
        ablation["epochs"],
        ablation["epoch_losses"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=f"Ablation (mask ratio={ablation['mask_ratio']})",
    )

    plt.title("Pretraining Average Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(outdir, "baseline_vs_ablation_loss.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_grad_norm_vs_step(baseline, ablation, outdir, smooth_window, clip_grad=None):
    base_norms = baseline["grad_norms"]
    abl_norms = ablation["grad_norms"]
    if clip_grad is not None:
        base_norms = np.clip(base_norms, 0, clip_grad)
        abl_norms = np.clip(abl_norms, 0, clip_grad)

    base_x = moving_average_x(baseline["global_steps"], smooth_window)
    base_y = moving_average(base_norms, smooth_window)

    abl_x = moving_average_x(ablation["global_steps"], smooth_window)
    abl_y = moving_average(abl_norms, smooth_window)

    plt.figure(figsize=(10, 6))
    plt.plot(
        base_x,
        base_y,
        linewidth=2,
        label=f"Baseline (mask ratio={baseline['mask_ratio']})",
    )
    plt.plot(
        abl_x,
        abl_y,
        linewidth=2,
        label=f"Ablation (mask ratio={ablation['mask_ratio']})",
    )

    if clip_grad is not None:
        plt.axhline(
            y=clip_grad,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Clip threshold ({clip_grad})",
        )
        title = f"Smoothed Gradient Norm vs Global Step (window={smooth_window}, clipped at {clip_grad})"
    else:
        title = f"Smoothed Gradient Norm vs Global Step (window={smooth_window})"

    plt.title(title)
    plt.xlabel("Global Step")
    plt.ylabel("Gradient Norm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(outdir, "baseline_vs_ablation_grad_norm.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_final_loss_bar(baseline, ablation, outdir, last_k):
    baseline_last_k = float(np.mean(baseline["epoch_losses"][-last_k:]))
    ablation_last_k = float(np.mean(ablation["epoch_losses"][-last_k:]))

    labels = [
        f"Baseline\n(mask={baseline['mask_ratio']})",
        f"Ablation\n(mask={ablation['mask_ratio']})",
    ]
    values = [baseline_last_k, ablation_last_k]

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

    plt.title(f"Mean Training Loss Over Last {last_k} Epochs")
    plt.ylabel("Loss")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(outdir, "baseline_vs_ablation_final_loss_bar.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_summary_csv(baseline, ablation, outdir, last_k):
    rows = []

    for run_name, run_data in [("baseline", baseline), ("ablation", ablation)]:
        rows.append(
            {
                "run": run_name,
                "mask_ratio": run_data["mask_ratio"],
                "final_epoch_loss": float(run_data["epoch_losses"][-1]),
                f"mean_last_{last_k}_epochs_loss": float(
                    np.mean(run_data["epoch_losses"][-last_k:])
                ),
            }
        )

    csv_path = os.path.join(outdir, "baseline_vs_ablation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-rank0", required=True)
    parser.add_argument("--baseline-rank1", required=True)
    parser.add_argument("--ablation-rank0", required=True)
    parser.add_argument("--ablation-rank1", required=True)
    parser.add_argument("--outdir", default="pretrain_ablation_plots")
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--last-k", type=int, default=5)
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        help="Simulate gradient clipping at this threshold before smoothing (e.g. 1.0). "
             "Clips displayed grad norms to show what training would look like with clipping applied.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    baseline = combine_rank_logs(args.baseline_rank0, args.baseline_rank1)
    ablation = combine_rank_logs(args.ablation_rank0, args.ablation_rank1)

    print(f"Baseline mask ratio: {baseline['mask_ratio']}")
    print(f"Ablation mask ratio: {ablation['mask_ratio']}")

    plot_loss_vs_epoch(baseline, ablation, args.outdir)
    plot_grad_norm_vs_step(baseline, ablation, args.outdir, args.smooth_window, args.clip_grad)
    plot_final_loss_bar(baseline, ablation, args.outdir, args.last_k)
    save_summary_csv(baseline, ablation, args.outdir, args.last_k)

    print(f"Plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()