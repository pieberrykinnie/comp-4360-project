import re
import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# this is the function that does the regex to parse the log file
# and it returns a list of dictionaries, where each dictionary corresponds to one log line with all the extracted metrics.
def parse_log_file(log_path):
    train_pattern = re.compile(
        r"Train:\s+\[(\d+)/(\d+)\]\[(\d+)/(\d+)\]\s+"
        r"eta\s+\S+\s+"
        r"lr\s*([0-9.eE+-]+)\s+"
        r"time\s+([0-9.]+)\s+\(\s*([0-9.]+)\)\s+"
        r"loss\s+([0-9.]+)\s+\(\s*([0-9.]+)\)\s+"
        r"grad_norm\s+([0-9.eE+-]+)\s+\(\s*([0-9.eE+-]+)\)\s+"
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

#this function takes the list of records (dictionaries) and builds two lists: one for epochs and one for the corresponding average losses.
#It uses a dictionary to keep track of the last loss value for each epoch, ensuring that if there are multiple log lines for the same epoch, only the last one is used for plotting.
def build_epoch_summary(records):

    epoch_to_last_loss = {}

    for row in records:
        epoch = row["epoch"]
        epoch_to_last_loss[epoch] = row["loss_avg"]

    epochs = sorted(epoch_to_last_loss.keys())
    losses = [epoch_to_last_loss[e] for e in epochs]

    return epochs, losses


def save_csv(records, output_csv):
    fieldnames = list(records[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

#This function takes the epochs and their corresponding average losses and creates a line plot.
#  The x-axis represents the epoch number, while the y-axis represents the average training loss.
#  The plot is saved as a PNG file in the specified output path.
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

#this function creates a line plot of the learning rate against the global step.
#  The x-axis represents the global step, while the y-axis represents the learning rate. 
# The plot is saved as a PNG file in the specified output path.
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


def plot_grad_norm_vs_global_step(records, output_path):
    x = [row["global_step"] for row in records]

    # Using grad_avg makes the plot smoother and easier to read.
    # If you want the raw version, change grad_avg -> grad_val.
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
    parser = argparse.ArgumentParser(description="Plot graphs from SimMIM pretraining log.")
    parser.add_argument(
        "--log",
        type=str,
        default="log_rank0.txt",
        help="Path to the log file"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Folder where the plots will be saved"
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = parse_log_file(log_path)

    if not records:
        print("No training records were found in the log file.")
        return

    # Save extracted data too, just in case you want it for Excel later
    save_csv(records, outdir / "extracted_metrics.csv")

    epochs, epoch_losses = build_epoch_summary(records)

    plot_loss_vs_epoch(epochs, epoch_losses, outdir / "avg_training_loss_vs_epoch.png")
    plot_lr_vs_global_step(records, outdir / "learning_rate_vs_global_step.png")
    plot_grad_norm_vs_global_step(records, outdir / "grad_norm_vs_global_step.png")

    print(f"Done. Plots saved in: {outdir}")
    print("Files created:")
    print("- avg_training_loss_vs_epoch.png")
    print("- learning_rate_vs_global_step.png")
    print("- grad_norm_vs_global_step.png")
    print("- extracted_metrics.csv")


if __name__ == "__main__":
    main()