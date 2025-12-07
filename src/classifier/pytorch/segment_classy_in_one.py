import pandas as pd
import cv2
import torch
import numpy as np
from pathlib import Path
from model_test_whole_torch import TS_segmenter,TS_classifier
from models_torch import LTSModel,GTSRBModel,s_custom_model
from torchvision import transforms
from pathlib import Path
import csv

num_classes = 43
img_width = 60
img_height = 60

CLASS_REF_PATH = Path("/home/panda/projects/german-street-sign/src/classifier/class_ref.csv")

def ref_class_names(class_ref: Path):
    with class_ref.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        return {int(row["ClassId"]): row["Name"].strip() for row in reader}




# classification model


model_segmentation = TS_segmenter(0.8)

model = LTSModel(num_classes,img_height,img_width)
model_classifier = TS_classifier(model,0.8)

img_folder = Path("/home/panda/projects/german-street-sign/Data/raw_data/final_test_img")
print("predictions = ")


for img_path in sorted(img_folder.glob("*.jpg")):
    print(f"Processing {img_path.name}")
    result_segmentation = model_segmentation.do_prediction(img_path)

    for index, row in result_segmentation.iterrows():
        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])
        img = cv2.imread(str(img_path))
        
        
        cropped_img = img[ymin:ymax, xmin:xmax]
        cv2.imshow("cropped",cropped_img)
        cv2.waitKey(0)
        #cropped_img = np.expand_dims(cropped_img,axis=0)
        class_names = ref_class_names(CLASS_REF_PATH)
        #print(cropped_img.shape)
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(cropped_img)

        base = [transforms.Resize((60,60)),
                transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)]
        transform = transforms.Compose(base)
        
        proc_img = transform(img_tensor).unsqueeze(0)
        print("Running Classification")
        result_classification = model_classifier.do_prediction(proc_img)
        pred_id = result_classification.argmax(1).item()
        pred_label = class_names.get(pred_id, f"Unknown({pred_id})")
        
        print(result_classification.argmax(1))
        print(pred_label)

    





