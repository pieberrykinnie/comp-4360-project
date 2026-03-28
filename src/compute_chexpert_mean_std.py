import os
import csv
import random
import argparse

from PIL import Image
import torch
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to CheXpert train.csv")
    parser.add_argument("--root", required=True,
                        help="Folder that contains 'CheXpert-v1.0-small'")
    parser.add_argument("--n", type=int, default=5000,
                        help="How many images to sample")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--img-size", type=int,
                        default=224, help="Resize/crop size")
    args = parser.parse_args()

    random.seed(args.seed)

    rel_paths = []
    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "Path" not in reader.fieldnames:
            raise ValueError(
                f"CSV does not have a 'Path' column. Columns: {reader.fieldnames}")

        for row in reader:
            p = row["Path"]
            if p:
                rel_paths.append(p)

    if len(rel_paths) == 0:
        raise ValueError("No paths found in the CSV.")

    n = min(args.n, len(rel_paths))
    sample_paths = random.sample(rel_paths, n)

    tform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    total_sum = 0.0
    total_sumsq = 0.0
    total_pixels = 0
    skipped = 0

    for rp in sample_paths:
        rp = rp.replace("/", os.sep)
        full_path = os.path.normpath(os.path.join(args.root, rp))

        if not os.path.exists(full_path):
            skipped += 1
            continue

        try:
            img = Image.open(full_path).convert("L")
            x = tform(img)
        except Exception:
            skipped += 1
            continue

        x = x.to(torch.float64)

        total_sum += x.sum().item()
        total_sumsq += (x * x).sum().item()
        total_pixels += x.numel()

    if total_pixels == 0:
        raise RuntimeError(
            "No images were successfully loaded. Check your paths/root.")

    mean = total_sum / total_pixels

    var = (total_sumsq / total_pixels) - (mean * mean)

    if var < 0:
        var = 0.0

    std = var ** 0.5

    print(f"Requested sample size: {args.n}")
    print(f"Used images:           {n - skipped}")
    print(f"Skipped images:        {skipped}")
    print(f"Mean (grayscale):      {mean:.6f}")
    print(f"Std  (grayscale):      {std:.6f}")


if __name__ == "__main__":
    main()
