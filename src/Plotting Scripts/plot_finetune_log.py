import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_log_file(log_path, rank_name):
    """
    Read one fine-tuning log file and extract training records.
    """

    train_pattern = re.compile(
        r"Train:\s+\[(\d+)/(\d+)\]\[(\d+)/(\d+)\]\s+"
        r"eta\s+\S+\s+"
        r"lr\s*([0-9.eE+-]+)\s+"
        r"time\s+([0-9.]+)\s+\(([0-9.]+)\)\s+"
        r"loss\s+([0-9.]+)\s+\(([0-9.]+)\)\s+"
        r"grad_norm\s+([0-9.eE+-]+)\s+\(([0-9.eE+-]+)\)\s+"
        r"mem\s+(\d+)MB"
    )

    records = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = train_pattern.search(line)
            if not match:
                continue

            epoch = int(match.group(1))
            total_epochs = int(match.group(2))
            step = int(match.group(3))
            steps_per_epoch = int(match.group(4))

            lr = float(match.group(5))
            time_val = float(match.group(6))
            time_avg = float(match.group(7))
            loss_val = float(match.group(8))
            loss_avg = float(match.group(9))
            grad_val = float(match.group(10))
            grad_avg = float(match.group(11))
            mem_mb = int(match.group(12))

            global_step = epoch * steps_per_epoch + step

            records.append({
                "rank": rank_name,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "steps_per_epoch": steps_per_epoch,
                "global_step": global_step,
                "lr": lr,
                "time_val": time_val,
                "time_avg": time_avg,
                "loss_val": loss_val,
                "loss_avg": loss_avg,
                "grad_val": grad_val,
                "grad_avg": grad_avg,
                "mem_mb": mem_mb,
            })

    return records

# this func combines records from multiple ranks by averaging the values for the same epoch and step.
def combine_records(records):

    grouped = defaultdict(list)

    for row in records:
        key = (row["epoch"], row["step"])
        grouped[key].append(row)

    combined = []

    for key in sorted(grouped.keys()):
        rows = grouped[key]

        combined_row = {
            "epoch": rows[0]["epoch"],
            "total_epochs": rows[0]["total_epochs"],
            "step": rows[0]["step"],
            "steps_per_epoch": rows[0]["steps_per_epoch"],
            "global_step": rows[0]["global_step"],
            "lr": sum(r["lr"] for r in rows) / len(rows),
            "time_val": sum(r["time_val"] for r in rows) / len(rows),
            "time_avg": sum(r["time_avg"] for r in rows) / len(rows),
            "loss_val": sum(r["loss_val"] for r in rows) / len(rows),
            "loss_avg": sum(r["loss_avg"] for r in rows) / len(rows),
            "grad_val": sum(r["grad_val"] for r in rows) / len(rows),
            "grad_avg": sum(r["grad_avg"] for r in rows) / len(rows),
            "mem_mb": sum(r["mem_mb"] for r in rows) / len(rows),
            "num_ranks_used": len(rows),
        }

        combined.append(combined_row)

    return combined

# i am tring to build a summary of the average training loss at the end of each epoch. 
# i  combine records and creates two lists: one for epochs and one for the corresponding average losses. 
# it uses a dictionary to keep track of the last loss value for each epoch, then sorts the epochs and extracts the losses in order.
def build_epoch_summary(records):
    epoch_to_last_loss = {}

    for row in records:
        epoch_to_last_loss[row["epoch"]] = row["loss_avg"]

    epochs = sorted(epoch_to_last_loss.keys())
    losses = [epoch_to_last_loss[e] for e in epochs]

    return epochs, losses


def save_csv(records, output_csv):
    if not records:
        return

    fieldnames = list(records[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

# i am ploting the avg training loss against epochs.
#  it creates a line plot with epochs on the x-axis and average training loss on the y-axis, 
def plot_loss_vs_epoch(epochs, losses, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Average Training Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# this func plots the learning rate against the global step
# it should create a line plit with global step on they x acis and learning rate on the y axis.
def plot_lr_vs_global_step(records, output_path):
    x = [row["global_step"] for row in records]
    y = [row["lr"] for row in records]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel("Global Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs Global Step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# this func plots the gradient norm against the global step.
def plot_grad_norm_vs_global_step(records, output_path):
    x = [row["global_step"] for row in records]
    y = [row["grad_avg"] for row in records]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel("Global Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm vs Global Step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot graphs from fine-tuning logs."
    )

    parser.add_argument(
        "--log0",
        type=str,
        default=r"output/simmim_finetune\simmim_finetune__vit_base__img224__100ep\log_rank0.txt",
        help="Path to log_rank0.txt"
    )

    parser.add_argument(
        "--log1",
        type=str,
        default=r"output/simmim_finetune\simmim_finetune__vit_base__img224__100ep\log_rank1.txt",
        help="Path to log_rank1.txt"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=r"output/simmim_finetune\simmim_finetune__vit_base__img224__100ep\plots",
        help="Folder where the plots will be saved"
    )

    args = parser.parse_args()

    log0_path = Path(args.log0)
    log1_path = Path(args.log1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_records = []

    if log0_path.exists():
        all_records.extend(parse_log_file(log0_path, "rank0"))
    else:
        print(f"Could not find: {log0_path}")

    if log1_path.exists():
        all_records.extend(parse_log_file(log1_path, "rank1"))
    else:
        print(f"Could not find: {log1_path}")

    if not all_records:
        print("No training records were found in either log file.")
        return

    combined_records = combine_records(all_records)

    if not combined_records:
        print("No combined records could be created.")
        return

    save_csv(combined_records, outdir / "combined_finetune_metrics.csv")

    epochs, epoch_losses = build_epoch_summary(combined_records)

    plot_loss_vs_epoch(epochs, epoch_losses, outdir / "avg_training_loss_vs_epoch.png")
    plot_lr_vs_global_step(combined_records, outdir / "learning_rate_vs_global_step.png")
    plot_grad_norm_vs_global_step(combined_records, outdir / "grad_norm_vs_global_step.png")

    print(f"Done. Plots saved in: {outdir}")
    print("Files created:")
    print("- avg_training_loss_vs_epoch.png")
    print("- learning_rate_vs_global_step.png")
    print("- grad_norm_vs_global_step.png")
    print("- combined_finetune_metrics.csv")


if __name__ == "__main__":
    main()