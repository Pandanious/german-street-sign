import json
from pathlib import Path
import cv2



img_dir = Path("/home/panda/projects/german-street-sign/Data/processed_data/mask_split/val/images")
label_dir = Path("/home/panda/projects/german-street-sign/Data/processed_data/mask_split/val/labels")
out_json = "/home/panda/projects/german-street-sign/Data/processed_data/mask_split/val/val_coco.json"

class_name = ["street_sign"]
categories = [{"id":0,"name": "street_sign"}]

images = []
annotations = []
ann_id = 1
img_id = 1


for img_path in sorted(img_dir.glob("*.*")):
    if img_path.suffix.lower() not in [".jpg",".png"]:
        continue

    img = cv2.imread(str(img_path))
    h,w = img.shape[:2]

    images.append({"id": img_id,"file_name":img_path.name,"width":w,"height":h})

    label_path = label_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            cls,cx,cy,bw,bh = map(float,line.split())
            x = (cx - bw / 2.0)*w
            y = (cy - bh / 2.0)*h
            img_w = bw * w
            img_h = bh * h

            annotations.append({"id":ann_id,"image_id":img_id,"category_id":int(cls),"bbox":[x,y,img_w,img_h],"area":img_h*img_w,"iscrowd":0})
        ann_id += 1
    img_id += 1 

coco = {"images": images,"annotations":annotations,"categories":categories}

Path(out_json).write_text(json.dumps(coco, indent=2))    
