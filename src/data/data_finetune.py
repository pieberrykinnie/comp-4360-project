import os
import csv
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms


# Same grayscale stats you computed from CheXpert
CHEXPERT_GRAY_MEAN = 0.533430
CHEXPERT_GRAY_STD = 0.283167


# These are the 14 CheXpert label columns we will predict.
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def parse_chexpert_label(value, uncertainty_policy="zeros"):
    """
    Convert one CSV label value into a float.

    CheXpert label values can be:
    - 1.0  -> positive
    - 0.0  -> negative
    - -1.0 -> uncertain
    - blank -> missing

    For now we keep uncertainty handling simple:
    - "zeros": treat -1 as 0
    - "ones": treat -1 as 1
    """

    if value is None:
        return 0.0

    value = str(value).strip()
    if value == "":
        return 0.0

    x = float(value)

    if x == -1.0:
        if uncertainty_policy == "ones":
            return 1.0
        return 0.0

    return x


class CheXpertFineTuneDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_root,
        transform,
        uncertainty_policy="zeros",
        frontal_only=True,
    ):
        self.transform = transform
        self.samples = []
        self.uncertainty_policy = uncertainty_policy

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            if "Path" not in reader.fieldnames:
                raise ValueError(f"'Path' column not found in CSV: {csv_path}")

            for label_name in CHEXPERT_LABELS:
                if label_name not in reader.fieldnames:
                    raise ValueError(f"Missing label column '{label_name}' in CSV: {csv_path}")

            for row in reader:
                if frontal_only:
                    if row.get("Frontal/Lateral", "").strip() != "Frontal":
                        continue

                rel_path = row["Path"].replace("/", os.sep)
                full_path = os.path.normpath(os.path.join(img_root, rel_path))

                if not os.path.exists(full_path):
                    continue

                target = []
                for label_name in CHEXPERT_LABELS:
                    target.append(
                        parse_chexpert_label(
                            row[label_name],
                            uncertainty_policy=self.uncertainty_policy
                        )
                    )

                self.samples.append((full_path, target))

        if len(self.samples) == 0:
            raise ValueError(
                f"No fine-tuning samples were loaded from {csv_path}. "
                f"Check csv_path, img_root, and whether frontal_only removed everything."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]

        try:
            img = Image.open(img_path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {img_path}") from e

        img_tensor = self.transform(img)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return img_tensor, target_tensor


def build_finetune_transform(is_train, config):

    mean = [CHEXPERT_GRAY_MEAN]
    std = [CHEXPERT_GRAY_STD]

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=config.DATA.IMG_SIZE,
                scale=(0.67, 1.0),
                ratio=(0.75, 1.333),
            ),
            # We keep horizontal flip off for chest X-rays
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(config.DATA.IMG_SIZE),
            transforms.CenterCrop(config.DATA.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transform


def build_dataset_finetune(is_train, config, logger=None):
    if is_train:
        csv_path = config.DATA.TRAIN_CSV_PATH
    else:
        csv_path = config.DATA.VAL_CSV_PATH

    img_root = config.DATA.IMG_ROOT
    transform = build_finetune_transform(is_train, config)

    dataset = CheXpertFineTuneDataset(
        csv_path=csv_path,
        img_root=img_root,
        transform=transform,
        uncertainty_policy=config.DATA.UNCERTAINTY_POLICY,
        frontal_only=config.DATA.FRONTAL_ONLY,
    )

    if logger is not None:
        logger.info(
            f"Built {'train' if is_train else 'val'} fine-tuning dataset "
            f"with {len(dataset)} samples"
        )
        logger.info(f"Transform (is_train={is_train}): {transform}")

    return dataset, len(CHEXPERT_LABELS)


def build_loader_finetune(config, logger=None):

    dataset_train, num_classes = build_dataset_finetune(True, config, logger)
    dataset_val, _ = build_dataset_finetune(False, config, logger)

    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes
    config.freeze()

    if logger is not None:
        logger.info(
            f"Build fine-tuning datasets: train={len(dataset_train)}, val={len(dataset_val)}"
        )
        logger.info(f"Number of classes set to {config.MODEL.NUM_CLASSES}")

    if dist.is_available() and dist.is_initialized():
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
    else:
        num_replicas = 1
        rank = 0

    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
    )

    sampler_val = DistributedSampler(
        dataset_val,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=False,
    )

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    mixup_fn = None

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn