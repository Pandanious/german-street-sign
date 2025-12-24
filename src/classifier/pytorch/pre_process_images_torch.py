import pathlib
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset,DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode


class GTSRBDataset(Dataset):
    def __init__(self, samples, class_to_idx, augment=False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.num_classes = len(self.class_to_idx)
        size = 60

        resize_largest_edge = transforms.Lambda(
                            lambda img: F.resize(
                                img,
                                (int(round(img.height * size / max(img.size))),
                                int(round(img.width  * size / max(img.size)))),
                                interpolation=InterpolationMode.BILINEAR))
        base = [
                resize_largest_edge,
                transforms.Lambda(lambda img: F.pad(
                    img,
                    ((size - img.size[0]) // 2, (size - img.size[1]) // 2,
                    (size - img.size[0] + 1) // 2, (size - img.size[1] + 1) // 2),
                    fill=0,
                )),
                    
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)]
        if augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=(0.9,1.1)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.05,0.05)),
                *base,
                ])
        else:
            self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x,target



def pre_process(batchsize = 64 , num_workers = 4, val_split = 0.2 , seed = 50):                        # num workers for sub processes (GPU, should be okay with 4. CPU ONLY change to 2 or default (0))

    train_dir = "Data/raw_data/Train"
    test_dir = "Data/raw_data/Test"

    for_split = datasets.ImageFolder(train_dir)
    total = len(for_split)
    val_n = int(total * val_split)
    train_n = total - val_n

    gen = torch.Generator().manual_seed(seed)

    train_split,val_split = random_split(range(total),[train_n,val_n], generator=gen)

    train_sample = [for_split.samples[i] for i in train_split.indices]
    val_samples = [for_split.samples[i] for i in val_split.indices]

    train_ds = GTSRBDataset(train_sample, for_split.class_to_idx, augment=True)
    val_ds = GTSRBDataset(val_samples, for_split.class_to_idx, augment=False)
   
    train_loader = DataLoader(train_ds, batch_size=batchsize,
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batchsize,
                            shuffle=False, num_workers=num_workers)
    test_loader = None

    return train_loader, val_loader, test_loader
        

        

        
