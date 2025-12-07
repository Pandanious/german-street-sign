import pathlib
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class GTSRBDataset(Dataset):
    def __init__(self,df, root_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.root_dir = pathlib.Path(root_dir)
        self.class_ids = sorted(self.df["ClassId"].astype(int).unique())
        self.num_classes = len(self.class_ids)
        base = [
            transforms.Resize((60,60)),
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
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(self.root_dir / row["Path"]).convert("RGB")
        x = self.transform(img)
        y = int(row["ClassId"])
        return x,y
    
def pre_process(batchsize = 64 , num_workers = 4 ):                        # num workers for sub processes (GPU, should be okay with 4. CPU ONLY change to 2 or default (0))

    meta = pd.read_csv("Data/raw_data/post_split.csv")
    train_df = meta[meta.split == "train"].copy()
    val_df = meta[meta.split == "val"].copy()
    test_df = pd.read_csv("Data/raw_data/Test.csv").copy()

    train_ds = GTSRBDataset(train_df, "Data/raw_data", augment=True)
    val_ds = GTSRBDataset(val_df, "Data/raw_data", augment=False)
    test_ds = GTSRBDataset(test_df, "Data/raw_data", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batchsize,
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batchsize,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batchsize,
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
        

        

        
