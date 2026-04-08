import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

# this is the func that help with parsing the log files and extracting the relevant metrics for plotting. 

def parse_train_lines(log_path, rank_name):
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

# this func is responsible for parsing the validation lines from the log file and extracting the mean AUC and validation loss for each epoch. 

def parse_validation_lines(log_path):

    val_pattern = re.compile(
        r"Validation - Epoch\s+(\d+):\s+mean_auc:\s+([0-9.]+)\s+\|\s+loss:\s+([0-9.]+)"
    )

    val_records = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = val_pattern.search(line)
            if not match:
                continue

            epoch = int(match.group(1))
            mean_auc = float(match.group(2))
            val_loss = float(match.group(3))

            val_records[epoch] = {
                "epoch": epoch,
                "mean_auc": mean_auc,
                "val_loss": val_loss,
            }

    return val_records


def combine_train_records(records):

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


def build_training_loss_by_epoch(records):
    epoch_to_last_loss = {}

    for row in records:
        epoch_to_last_loss[row["epoch"]] = row["loss_avg"]

    epochs = sorted(epoch_to_last_loss.keys())
    losses = [epoch_to_last_loss[e] for e in epochs]

    return epochs, losses


def build_validation_lists(val_records):
    epochs = sorted(val_records.keys())
    val_losses = [val_records[e]["val_loss"] for e in epochs]
    mean_aucs = [val_records[e]["mean_auc"] for e in epochs]

    return epochs, val_losses, mean_aucs


def save_csv(records, output_csv):
    if not records:
        return

    fieldnames = list(records[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_validation_csv(val_records, output_csv):
    rows = [val_records[e] for e in sorted(val_records.keys())]

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_loss(epochs, losses, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_validation_loss(epochs, losses, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mean_auroc(epochs, aucs, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, aucs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean AUROC")
    plt.title("Mean AUROC vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot fine-tuning graphs from rank0 and rank1 logs."
    )

    parser.add_argument(
        "--log0",
        type=str,
        default=r"output\simmim_finetune\simmim_finetune__vit_base__img224__100ep\log_rank0.txt",
        help="Path to log_rank0.txt"
    )

    parser.add_argument(
        "--log1",
        type=str,
        default=r"output\simmim_finetune\simmim_finetune__vit_base__img224__100ep\log_rank1.txt",
        help="Path to log_rank1.txt"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=r"output/simmim_finetune\simmim_finetune__vit_base__img224__100ep\plots",
        help="Folder where plots will be saved"
    )

    args = parser.parse_args()

    log0_path = Path(args.log0)
    log1_path = Path(args.log1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_train_records = []

    if log0_path.exists():
        all_train_records.extend(parse_train_lines(log0_path, "rank0"))
    else:
        print(f"Could not find: {log0_path}")

    if log1_path.exists():
        all_train_records.extend(parse_train_lines(log1_path, "rank1"))
    else:
        print(f"Could not find: {log1_path}")

    if not all_train_records:
        print("No training records were found.")
        return

    combined_train_records = combine_train_records(all_train_records)


    if log0_path.exists():
        validation_records = parse_validation_lines(log0_path)
    else:
        validation_records = {}

    if not validation_records:
        print("No validation records were found in log_rank0.txt.")
        return

    save_csv(combined_train_records, outdir / "combined_finetune_train_metrics.csv")
    save_validation_csv(validation_records, outdir / "finetune_validation_metrics.csv")

    train_epochs, train_losses = build_training_loss_by_epoch(combined_train_records)
    val_epochs, val_losses, mean_aucs = build_validation_lists(validation_records)

    plot_training_loss(train_epochs, train_losses, outdir / "training_loss_vs_epoch.png")
    plot_validation_loss(val_epochs, val_losses, outdir / "validation_loss_vs_epoch.png")
    plot_mean_auroc(val_epochs, mean_aucs, outdir / "mean_auroc_vs_epoch.png")

    print(f"Done. Plots saved in: {outdir}")
    print("Files created:")
    print("- training_loss_vs_epoch.png")
    print("- validation_loss_vs_epoch.png")
    print("- mean_auroc_vs_epoch.png")
    print("- combined_finetune_train_metrics.csv")
    print("- finetune_validation_metrics.csv")


if __name__ == "__main__":
    main()