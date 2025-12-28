import pandas as pd
import cv2
import torch
import numpy as np
import PIL
from pathlib import Path
from model_test_whole_torch import TS_segmenter,TS_classifier
from models_torch import LTSModel,GTSRBModel,s_custom_model
from torchvision import transforms
import torchvision.ops as ops
import torch.nn.functional as F
from pathlib import Path
import csv
import os

num_classes = 54
img_width = 60
img_height = 60
det_num = 0

info_file = Path("/home/panda/projects/german-street-sign/Results/res_video.txt") 
os.makedirs(info_file.parent, exist_ok=True)

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

def iou_box(box,boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    w  = np.maximum(0.0, x2 - x1)
    h  = np.maximum(0.0, y2 - y1)
    intersection_area = w * h

    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = ((boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]))
    union = box_area + boxes_area - intersection_area
    return intersection_area / np.maximum(union,1e-6)

def wbf(df,iou_threshold):
    if df.empty:
        return df
    df = df.sort_values("confidence",ascending = False).reset_index(drop=True)
    used = np.zeros(len(df), dtype=bool)
    fused_rows = []
    
    for i in range(len(df)):
        if used[i]:
            continue
        box_i = df.loc[i,["xmin","ymin","xmax","ymax"]].to_numpy(dtype=np.float32)
        cand = []
        scores = []

        for j in range(i,len(df)):
            if used[j]:
                continue
            
            box_j = df.loc[j,["xmin","ymin","xmax","ymax"]].to_numpy(dtype=np.float32)
            iou = iou_box(box_i,box_j[None, :])[0]
            if iou >= iou_threshold:
                used[j] = True
                cand.append(box_j)
                scores.append(float(df.loc[j, "confidence"]))
        
        weights = np.array(scores, dtype=np.float32)
        cand = np.array(cand, dtype=np.float32)
        fused_box = (cand * weights[:,None]).sum(axis = 0) / weights.sum()
        fused_rows.append({"xmin": float(fused_box[0]),"ymin": float(fused_box[1]),"xmax": float(fused_box[2]),"ymax": float(fused_box[3]),"confidence":float(np.max(weights))})

    return pd.DataFrame(fused_rows)



def do_tiling_det(model, frame, tile_x=640, tile_y = 640 , overlap= 0.3, pad = 100):

    bgr_img = frame
    if bgr_img is None:
        return pd.DataFrame(columns=["xmin","ymin","xmax","ymax","confidence"])
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h,w = rgb_img.shape[:2]

    stride_x = int(tile_x * (1-overlap))
    #stride_x = max(1, stride_x)
    stride_y = int(tile_y * (1-overlap))
    #stride_y = max(1, stride_y)


    img_df = []

    for y in range(0,h,stride_y):
        for x in range(0,w,stride_x):
            x_orig = x
            y_orig = y
            y1 = min(y + tile_y,h)
            x1 = min(x + tile_x,w)

            pad_x_orig = max(0,x_orig - pad)
            pad_y_orig = max(0,y_orig - pad)
            pad_x1 = min(w,x1 + pad)
            pad_y1 = min(h,y1 + pad)

            tile_img = rgb_img[pad_y_orig:pad_y1,pad_x_orig:pad_x1]

            if tile_img.size == 0:
                continue

            with torch.inference_mode():
                results = model(tile_img)
            
            df = results.pandas().xyxy[0]
            if df.empty:
               continue


            df = df.copy()
            df["xmin"] += pad_x_orig
            df["xmax"] += pad_x_orig
            df["ymin"] += pad_y_orig
            df["ymax"] += pad_y_orig

            cx = (df["xmin"] + df["xmax"]) / 2
            cy = (df["ymin"] + df["ymax"]) / 2
            retain = (cx >= x_orig) & (cx <= x1) & (cy >= y_orig) & (cy <= y1)
            df = df[retain]

            if not df.empty:
                img_df.append(df)

    if not img_df:
        return pd.DataFrame(columns=["xmin","ymin","xmax","ymax","confidence"])
    
    merged = pd.concat(img_df, ignore_index=True)
    merged = wbf(merged,iou_threshold=0.1)

    return merged

def helper_box(img,box_text, margin=20, padding=10,
               font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.2, thickness=1,
               text_color=(0, 255, 255), box_color=(0, 0, 0), line_gap=6):
    if not box_text:
        return img
    sizes = [cv2.getTextSize(t,font,font_scale,thickness) for t in box_text]
    max_w = max(s[0][0] for s in sizes)
    line_h = max(s[0][1] for s in sizes)
    box_h = len(box_text) * (line_h + line_gap) - line_gap + 2 * padding
    box_w = max_w + 2 * padding
    h, w = img.shape[:2]

    x1 = margin
    y2 = h - margin
    y1 = max(0, y2 - box_h)
    cv2.rectangle(img, (x1, y1), (x1 + box_w, y2), box_color, -1)
    y = y2 - padding
    for text in reversed(box_text):
        cv2.putText(img, text, (x1 + padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y -= line_h + line_gap






####################################################################################################################################################
# detection and classification start here!!! 
####################################################################################################################################################

model_segmentation = TS_segmenter(0.5)

model = s_custom_model(num_classes,img_height,img_width)
model_classifier = TS_classifier(model,0.9)
data_folder = Path("/home/panda/projects/german-street-sign/Data/test")



with open(info_file,'a') as f:
    f.write("predictions\n")

result_dir = Path("/home/panda/projects/german-street-sign/Results")
result_dir.mkdir(parents=True,exist_ok=True)


for img_path in sorted(data_folder.glob("*.mp4")):
    with open(info_file,'a') as f:
        f.write("-------------------------------------------------------------------------------------------------------\n")
        f.write(f"Processing {img_path.name}\n")
        f.write("\n")
    cap = cv2.VideoCapture(str(img_path))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(result_dir / f"{img_path.stem}_annotated.mp4")

    writer  = None

    while True:
        ret,frame = cap.read()        
        if not ret:
           print("End of video")
           break
        fh,fw = frame.shape[:2]
        print(fh,fw)

        if writer is None:
            writer = cv2.VideoWriter(out_path,fourcc, cap_fps ,(fw,fh))

    #result_segmentation = model_segmentation.do_detection(img_path)
        result_detection = do_tiling_det(model_segmentation.model, frame, tile_x = fw//4, tile_y = fh//4, overlap = 0.2, pad=100 )
        det_num = 0
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box_texts = []

        for index, row in result_detection.iterrows():
            det_num += 1
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            
            detector_prob = float(row['confidence'])
            with open(info_file,'a') as f:
                f.write("Running Detection\n")
                f.write(f"Probability of {det_num}: {detector_prob}\n")
                f.write(f"Bounding Box: x1: {xmin}, y1: {ymin}, x2: {xmax}, y2: {ymax}\n")
                

            #text_org = (xmin+1, int((ymax+ymin)/2))
            text_org = xmin+10, ymin
            
            cropped_img = img[ymin:ymax, xmin:xmax]
            #print(cropped_img.shape)
            
            #cropped_img = np.expand_dims(cropped_img,axis=0)
            class_names = ref_class_names(CLASS_REF_PATH)
            #print(cropped_img.shape)
            transform = transforms.Compose([transforms.ToTensor()])
            img_tensor = transform(cropped_img)                                                                            # Padding to mantain aspect ration while resizing.
            img_tensor = resize_with_aspect(img_tensor,target=img_height)                                                  #
            transform = transforms.Compose([transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)])
            
            proc_img = transform(img_tensor).unsqueeze(0)
            with open(info_file,'a') as f:
                f.write("Running Classification\n")
            result_classification = model_classifier.do_classification(proc_img)
            prob_vals, class_ids = result_classification.max(dim=1)
            pred_id = class_ids.item()
            pred_prob = prob_vals.item()
            pred_label = class_names.get(pred_id, f"Unknown({pred_id})")
            if pred_id != 53:
                if pred_prob >= .70:
                    with open(info_file,'a') as f:
                        f.write(f"Class No. {pred_id}: {pred_label}, Probability: {pred_prob}\n")

                    #res_text = f"{pred_id}: {pred_label}"
                    res_text = f"{pred_id}"
                    box_text = f"{pred_id}: {pred_label}, Probability: {pred_prob:.2f}"
                    box_texts.append(box_text)
                    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)
                #print("res_text ", res_text, "Prob: ",pred_prob )
                    cv2.putText(   img,
                            res_text,
                            text_org,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,)
                    helper_box(img,box_texts)

                else:
                    continue
            else:
                continue
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)

    if writer is not None:
        writer.release()
    cap.release()



    





