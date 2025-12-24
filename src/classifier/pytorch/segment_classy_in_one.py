import pandas as pd
import cv2
import torch
import numpy as np
import PIL
from pathlib import Path
from model_test_whole_torch import TS_segmenter,TS_classifier
from models_torch import LTSModel,GTSRBModel,s_custom_model
from torchvision import transforms
import torch.nn.functional as F
from pathlib import Path
import csv

num_classes = 51
img_width = 60
img_height = 60

CLASS_REF_PATH = Path("/home/panda/projects/german-street-sign/src/classifier/class_ref.csv")

def ref_class_names(class_ref: Path):
    with class_ref.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        return {int(row["ClassId"]): row["Name"].strip() for row in reader}
    
def resize_with_aspect(img_tensor, target = 60):
    c,h,w = img_tensor.shape
    scale = target / max(h,w)
    new_h, new_w = int(round(h*scale)), int(round(w*scale))
    pad_h = target - new_h
    pad_w = target - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    img = F.interpolate(img_tensor.unsqueeze(0), size=(new_h,new_w), mode="bilinear",align_corners=False)
    img = F.pad(img,(pad_left,pad_right,pad_top,pad_bottom), value=.5)

    return img.squeeze(0)






# classification model


model_segmentation = TS_segmenter(0.5)

model = s_custom_model(num_classes,img_height,img_width)
model_classifier = TS_classifier(model,0.6)
img_folder = Path("/home/panda/projects/german-street-sign/Data/raw_data/final_test_img")
print("predictions = ")
result_dir = Path("/home/panda/projects/german-street-sign/Results")

for img_path in sorted(img_folder.glob("*.jpg")):
    print("-------------------------------------------------------------------------------------------------------")
    print("\n")
    print(f"Processing {img_path.name}")
    print("-------------------------------------------------------------------------------------------------------")
    result_segmentation = model_segmentation.do_prediction(img_path)
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for index, row in result_segmentation.iterrows():
        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])
        

        #text_org = (xmin+1, int((ymax+ymin)/2))
        text_org = (xmin, ymax+50)
        
        cropped_img = img[ymin:ymax, xmin:xmax]
        print(cropped_img.shape)
        
        #cropped_img = np.expand_dims(cropped_img,axis=0)
        class_names = ref_class_names(CLASS_REF_PATH)
        #print(cropped_img.shape)
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(cropped_img)                                                                            # Padding to mantain aspect ration while resizing.
        img_tensor = resize_with_aspect(img_tensor,target=img_height)                                                  #
        transform = transforms.Compose([transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)])
        
        proc_img = transform(img_tensor).unsqueeze(0)
        print("Running Classification")
        result_classification = model_classifier.do_prediction(proc_img)
        prob_vals, class_ids = result_classification.max(dim=1)
        pred_id = class_ids.item()
        pred_prob = prob_vals.item()
        pred_label = class_names.get(pred_id, f"Unknown({pred_id})")
        if pred_prob >= .70:
        
            print(f"Class No. {pred_id}: {pred_label}, Probability: {pred_prob}")
            res_text = f"{pred_id}: {pred_label}"
            
        
        else:
            res_text = "Unknown Sign"

        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),8)
        print("res_text ", res_text, "Prob: ",pred_prob )
        cv2.putText(   img,
                    res_text,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA,)
        
    resized_down = cv2.resize(img, None, fx=.4, fy=.4, interpolation=cv2.INTER_LINEAR)
    resized_down = cv2.cvtColor(resized_down, cv2.COLOR_RGB2BGR)
    result_dir.mkdir(parents=True,exist_ok=True)
    out_path = result_dir / f"{img_path.stem}_annotated.jpg"
    cv2.imwrite(str(out_path),resized_down)

    


    





