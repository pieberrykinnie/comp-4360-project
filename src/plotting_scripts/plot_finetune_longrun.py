import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


TRAIN_PATTERN = re.compile(
    r"Train: \[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?"
    r"lr ([0-9.eE+-]+).*?"
    r"loss ([0-9.]+) \(([0-9.]+)\).*?"
    r"grad_norm ([0-9.]+) \(([0-9.]+)\)"
)

VAL_PATTERN = re.compile(
    r"Validation - Epoch (\d+): mean_auc: ([0-9.]+) \| loss: ([0-9.]+)"
)


def parse_log(log_path):
    train_rows = []
    val_rows = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            train_match = TRAIN_PATTERN.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                total_epochs = int(train_match.group(2))
                step = int(train_match.group(3))
                steps_per_epoch = int(train_match.group(4))
                lr = float(train_match.group(5))
                loss_current = float(train_match.group(6))
                loss_avg = float(train_match.group(7))
                grad_current = float(train_match.group(8))
                grad_avg = float(train_match.group(9))

                global_step = epoch * steps_per_epoch + step

                train_rows.append(
                    {
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "step": step,
                        "steps_per_epoch": steps_per_epoch,
                        "global_step": global_step,
                        "lr": lr,
                        "loss_current": loss_current,
                        "loss_avg": loss_avg,
                        "grad_current": grad_current,
                        "grad_avg": grad_avg,
                    }
                )

            val_match = VAL_PATTERN.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                mean_auroc = float(val_match.group(2))
                val_loss = float(val_match.group(3))

                val_rows.append(
                    {
                        "epoch": epoch,
                        "mean_auroc": mean_auroc,
                        "val_loss": val_loss,
                    }
                )

    return train_rows, val_rows


def combine_training_rows(rank0_rows, rank1_rows):
    grouped = {}

    for row in rank0_rows + rank1_rows:
        key = (row["epoch"], row["step"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    combined = []
    for key in sorted(grouped.keys()):
        rows = grouped[key]

        combined.append(
            {
                "epoch": rows[0]["epoch"],
                "step": rows[0]["step"],
                "global_step": int(np.mean([r["global_step"] for r in rows])),
                "lr": float(np.mean([r["lr"] for r in rows])),
                "loss_avg": float(np.mean([r["loss_avg"] for r in rows])),
                "grad_avg": float(np.mean([r["grad_avg"] for r in rows])),
            }
        )

    return combined


def combine_validation_rows(rank0_rows, rank1_rows):
    grouped = {}

    for row in rank0_rows + rank1_rows:
        epoch = row["epoch"]
        if epoch not in grouped:
            grouped[epoch] = []
        grouped[epoch].append(row)

    combined = []
    for epoch in sorted(grouped.keys()):
        rows = grouped[epoch]

        combined.append(
            {
                "epoch": epoch,
                "mean_auroc": float(np.mean([r["mean_auroc"] for r in rows])),
                "val_loss": float(np.mean([r["val_loss"] for r in rows])),
            }
        )

    return combined


def get_epoch_training_curve(combined_train_rows):
    epoch_to_rows = {}

    for row in combined_train_rows:
        epoch = row["epoch"]
        if epoch not in epoch_to_rows:
            epoch_to_rows[epoch] = []
        epoch_to_rows[epoch].append(row)

    epochs = []
    train_losses = []

    for epoch in sorted(epoch_to_rows.keys()):
        rows = sorted(epoch_to_rows[epoch], key=lambda x: x["step"])
        final_running_loss = rows[-1]["loss_avg"]

        epochs.append(epoch)
        train_losses.append(final_running_loss)

    return np.array(epochs), np.array(train_losses)


def get_validation_curves(combined_val_rows):
    epochs = []
    mean_aurocs = []
    val_losses = []

    for row in combined_val_rows:
        epochs.append(row["epoch"])
        mean_aurocs.append(row["mean_auroc"])
        val_losses.append(row["val_loss"])

    return np.array(epochs), np.array(mean_aurocs), np.array(val_losses)


def plot_training_loss(epochs, train_losses, outdir):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker="o", markersize=3)
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "training_loss_vs_epoch.png"), dpi=200)
    plt.close()


def plot_validation_loss(epochs, val_losses, outdir):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, marker="o", markersize=3)
    plt.title("Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "validation_loss_vs_epoch.png"), dpi=200)
    plt.close()


def plot_mean_auroc(epochs, mean_aurocs, outdir):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_aurocs, marker="o", markersize=3)
    plt.title("Mean AUROC vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean AUROC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_auroc_vs_epoch.png"), dpi=200)
    plt.close()


def plot_first_vs_last_100_auroc(mean_aurocs, outdir):
    first_100 = mean_aurocs[:100]
    last_100 = mean_aurocs[100:200]

    x_first = np.arange(len(first_100))
    x_last = np.arange(len(last_100))

    plt.figure(figsize=(10, 6))
    plt.plot(x_first, first_100, marker="o", markersize=3, label="First 100 epochs")
    plt.plot(x_last, last_100, marker="o", markersize=3, label="Last 100 epochs")
    plt.title("Mean AUROC: First 100 vs Last 100 Epochs")
    plt.xlabel("Epoch in Window")
    plt.ylabel("Mean AUROC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_auroc_first_vs_last_100.png"), dpi=200)
    plt.close()


def plot_first_vs_last_100_val_loss(val_losses, outdir):
    first_100 = val_losses[:100]
    last_100 = val_losses[100:200]

    x_first = np.arange(len(first_100))
    x_last = np.arange(len(last_100))

    plt.figure(figsize=(10, 6))
    plt.plot(x_first, first_100, marker="o", markersize=3, label="First 100 epochs")
    plt.plot(x_last, last_100, marker="o", markersize=3, label="Last 100 epochs")
    plt.title("Validation Loss: First 100 vs Last 100 Epochs")
    plt.xlabel("Epoch in Window")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "validation_loss_first_vs_last_100.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank0", required=True, help="Path to log_rank0.txt")
    parser.add_argument("--rank1", required=True, help="Path to log_rank1.txt")
    parser.add_argument("--outdir", default="only_finetune_plots", help="Folder to save plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rank0_train, rank0_val = parse_log(args.rank0)
    rank1_train, rank1_val = parse_log(args.rank1)

    combined_train = combine_training_rows(rank0_train, rank1_train)
    combined_val = combine_validation_rows(rank0_val, rank1_val)

    train_epochs, train_losses = get_epoch_training_curve(combined_train)
    val_epochs, mean_aurocs, val_losses = get_validation_curves(combined_val)

    plot_training_loss(train_epochs, train_losses, args.outdir)
    plot_validation_loss(val_epochs, val_losses, args.outdir)
    plot_mean_auroc(val_epochs, mean_aurocs, args.outdir)
    plot_first_vs_last_100_auroc(mean_aurocs, args.outdir)
    plot_first_vs_last_100_val_loss(val_losses, args.outdir)

    print(f"Saved 5 plots to: {args.outdir}")


if __name__ == "__main__":
    main()