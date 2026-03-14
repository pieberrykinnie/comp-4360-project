from asyncio.log import logger
from logging import config
import os
import csv 
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms


class MaskGenerator:
    def __init__(self, input_size, mask_patch_size, model_patch_size, mask_ratio):
        # We are validating the input parameters
        if input_size <= 0:
            raise ValueError("The input size needs to be a positive integer")
        if mask_patch_size <= 0 or model_patch_size <= 0:
            raise ValueError("Patch sizes need to be positive integers")
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError("the mask ratio must be between 0 and 1")
        if input_size % mask_patch_size != 0:
            raise ValueError("The input size must be divisible by the mask patch size")
        if mask_patch_size % model_patch_size != 0:
            raise ValueError("The mask patch size must be divisible by the model patch size")
        
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = float(mask_ratio)

        self.mask_grid = input_size // mask_patch_size
        self.model_grid = input_size // model_patch_size
        # we are calculating the scale factor between the mask patch size and 
        # the model patch size, which will be used to determine how many model
        #  patches fit into one mask patch
        self.scale = mask_patch_size // model_patch_size

        #how many blockks exist in total
        self.num_mask_blocks = self.mask_grid * self.mask_grid
        #how many we are actually masking
        self.num_to_mask = int(round(self.num_mask_blocks * self.mask_ratio))


    #returns a token_mask where each entry is typically 0 or 1, 
    # indicating whether the corresponding patch is masked (1) or not (0).    
    def __call__(self) -> np.ndarray:
        # i make a flat array of zeros so nothing is masked yet
        coarse_flat = np.zeros(self.num_mask_blocks, dtype=int)

        # we are randomly picking what blocks we want to mask
        if self.num_to_mask > 0:
            masked_idx = np.random.choice(self.num_mask_blocks, size=self.num_to_mask, replace=False)
            coarse_flat[masked_idx] = 1

        # this makes it a 2d coarse grid, so we make it a square matrix
        coarse_mask = coarse_flat.reshape(self.mask_grid, self.mask_grid)
        # we repeat the rows and columns of the coarse mask to create a token mask
        # that matches the models patch size.
        token_mask = np.repeat(np.repeat(coarse_mask, self.scale, axis=0), self.scale, axis=1)

        return token_mask



class SimMIMDataset:

    def __init__(self, img_size: int, in_chans: int, mask_generator, train: bool = True,
        use_random_resized_crop: bool = True, use_hflip: bool = False, mean=None, std=None):

        if img_size <= 0:
            raise ValueError("Eror the image size has to be positive.")
        if in_chans not in (1, 3):
            raise ValueError("the input channels must be either 1 or 3")
        
        self.img_size = img_size
        self.in_chans = in_chans
        self.mask_generator = mask_generator
        self.train = train

        if mean is None or std is None:
            if in_chans == 1:
                mean = [0.5]
                std = [0.5]
            # i got this fromt the imagenet dataset.
            # we are only doing greyscale but i thought it would be good to have the option
            # for 3 channels as well, so i just used the imagenet mean and std.
            else:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

        if len(mean) != in_chans or len(std) != in_chans:
                raise ValueError(f"the length of mean and std must match the number of input channels ({in_chans})")

        self.mean = mean
        self.std = std


        if train:
            if use_random_resized_crop:
                spatial = transforms.RandomResizedCrop(
                    size=img_size,
                    scale=(0.67, 1.0),   
                    ratio=(0.75, 1.333), 
                )
            else:
                spatial = transforms.Resize((img_size, img_size))
            
            #this means we do the spatial transform first
            ops = [spatial]

            #if we eneable this then we will randomly flip the image horizontally half the time.
            if use_hflip:
                ops.append(transforms.RandomHorizontalFlip(p=0.5))
            
            #convert the image to a tensor and normalize it 
            ops += [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
            # this will turn all our operations into a single function so that we can apply it to our images in the dataset class.
            self.img_transform = transforms.Compose(ops)

        # if we dont want any augmentation then we simply resize and cebtre the image and then convert it to a tensor and normalize it.
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

    # makes our SimMIMDataset act like a function
    def __call__(self, img):
        # so if we are doing greyscale the we force 1 channel
        if self.in_chans == 1:
            img = img.convert("L") 
            #if rgb then 3 chanels
        else:
            img = img.convert("RGB")
        
        # this is where i do the transofrmations that are in the __init__.
        img_tensor = self.img_transform(img)

        # this will generate our mask
        mask_np = self.mask_generator()
        # convert to torch because the model will need a torch tensor
        mask = torch.from_numpy(mask_np).long()

        return img_tensor, mask
    
# new dataset class 
class CheXpertPretrainDataset(Dataset):
     
     # transform is the SimMimTransform object
     # frontal_only is for if we want only front or bothe front abd back images. 
     # i put limit for debugging and testing so that we dont have to load the whole dataset
     #  every time, but it can be set to None to load everything.
    def __init__(self, csv_path, img_root, transform, frontal_only=True, limit=None):
        self.transform = transform
        self.samples = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:

                if frontal_only:

                    if row.get("Frontal/Lateral", "").strip() != "Frontal":
                        continue

                rel_path = row["Path"]
                rel_path = rel_path.replace("/", os.sep)
                full_path = os.path.normpath(os.path.join(img_root, rel_path))
                self.samples.append(full_path)
                
                if limit is not None and len(self.samples) >= limit:
                    break
            
        if len(self.samples) == 0:
            raise ValueError(
                "No images were loaded. Check:\n"
                "- csv_path is correct\n"
                "- img_root is correct (it should contain 'CheXpert-v1.0-small')\n"
                "- frontal_only isn't filtering everything"
        )

    def __len__(self):
        # How many samples we have in the dataaset
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {img_path}") from e
        
        img_tensor, mask_tensor = self.transform(img)
        dummy_target = 0
        return (img_tensor, mask_tensor), dummy_target
    
# This is for the dataloader to collate our samples into batches. 

def collate_fn(batch):

    #here i am splitting the batch into images, masks, and targets.
    imgs = [item[0][0] for item in batch]   
    masks = [item[0][1] for item in batch]  
    targets = [item[1] for item in batch]

    imgs = torch.stack(imgs, dim=0)
    masks = torch.stack(masks, dim=0)

    #now we convert the targets to a tensor, 
    targets = torch.tensor(targets, dtype=torch.long)

    return imgs, masks, targets
    

def build_loader_simmim(config, logger, is_train=True):
    
    csv_path = config.DATA.CSV_PATH
    img_root = config.DATA.IMG_ROOT

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"the CSV file was not found: {csv_path}")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"the image root directory was not found: {img_root}")
    
    if config.MODEL.TYPE.lower() == "vit":
        in_chans = config.MODEL.VIT.IN_CHANS
        model_patch_size = config.MODEL.VIT.PATCH_SIZE    
    else: 
        in_chains = config.MODEL.VIT.IN_CHANS
        model_patch_size = config.MODEL.VIT.PATCH_SIZE

    img_size = config.DATA.IMG_SIZE
    mask_patch_size = config.DATA.MASK_PATCH_SIZE
    mask_ratio = config.DATA.MASK_RATIO

    #now we build the mask generator and the transform
    mask_generator = MaskGenerator(
        input_size=img_size,
        mask_patch_size=mask_patch_size,
        model_patch_size=model_patch_size,
        mask_ratio=mask_ratio,
    )

    transform = SimMIMDataset(
        img_size=img_size,
        in_chans=in_chans,
        mask_generator=mask_generator,
        train=is_train,
        use_random_resized_crop=is_train,
        use_hflip=is_train,
    )

    # this is for debug n testing
    if logger is not None:
        logger.info(f"Using csv_path={csv_path}")
        logger.info(f"Using img_root={img_root}")
        logger.info(f"img_size={img_size}, in_chans={in_chans}, model_patch_size={model_patch_size}")
        logger.info(f"mask_patch_size={mask_patch_size}, mask_ratio={mask_ratio}")


    #this is the dataset and the loader
    dataset = CheXpertPretrainDataset(
        csv_path=csv_path,
        img_root=img_root,
        transform=transform,
        frontal_only=True  
    )

    if logger is not None:
        logger.info(f"Dataset size: {len(dataset)} samples")

    # this is the build sampler
    if dist.is_available() and dist.is_initialized():
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
    else:
        num_replicas = 1
        rank = 0
    
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=is_train)   

    # this is the dataloader
    dataloader = DataLoader(
         dataset,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    return dataloader