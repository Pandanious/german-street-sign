from pathlib import Path
import torch
import cv2
import numpy as np

class CustDetDataset(torch.utils.data.Dataset):
    def __init__(self,img_dir,label_dir,transform=None,img_extensions = (".jpg",".png")):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transform
        self.images = []
        for ext in img_extensions:
            self.images.extend(self.img_dir.glob(f"*.{ext}"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]

        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        if label_path.exists():
            for l in label_path.read_text().strip().splitlines():
                if not l:
                    continue
                cls, cx, cy, cw, ch = map(float, l.split())
                x1 = (cx - cw / 2.0 ) * w
                x2 = (cx + cw / 2.0 ) * w 
                y1 = (cy - ch / 2.0 ) * h
                y2 = (cy + ch / 2.0 ) * h
                boxes.append([x1,y1,x2,y2])
                labels.append(int(cls))

        target = {"boxes":torch.tensor(boxes, dtype=torch.float32),
                  "labels":torch.tensor(labels, dtype=torch.int64),
                  "image_id":torch.tensor([index]),
                  "img_orig_size":torch.tensor([h,w]),
                  "size":torch.tensor([h,w])}
        
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

class DetrTransform:
    def __init__(self,size = 640,max_size = 1000, mean=[0.485, 0.456, 0.406], std_dev =[0.229, 0.224, 0.225]):
        self.size = size
        self.max_size = max_size
        self.mean = np.array(mean, dtype = np.float32)
        self.std_dev = np.array(std_dev, dtype= np.float32)

    def __call__(self, img, target):
        h,w = img.shape[:2]
        scale = self.size / min(h,w)
        if max(h,w) * scale > self.max_size:
            scale = self.max_size / max(h,w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img_resize = cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_LINEAR)

        if target["boxes"].numel() > 0:
            boxes = target["boxes"]
            boxes = boxes * torch.tensor([scale,scale,scale,scale])
            target["boxes"] = boxes
        target["size"] =  torch.tensor([new_h,new_w])

        img = img_resize.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std_dev
        img = torch.from_numpy(img).permute(2,0,1).contiguous()

        return img, target
    
def detr_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

