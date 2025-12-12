import cv2
import numpy as np
import glob
prohibitory = 0
mandatory = 0
danger = 0
other = 0
a = np.array([0,0,0,0,0])
count = 0


#bbox_file_path = "/home/panda/projects/german-street-sign/Data/processed_data/mask_split/train/labels/*.txt"
bbox_file_path = "/home/panda/projects/german-street-sign/Data/processed_data/mask_split/val/labels/*.txt"




for name in glob.glob(bbox_file_path):
    bbox_open = np.loadtxt(name)
    bbox_open = np.atleast_2d(bbox_open)
    for row in bbox_open:
        if row.shape != a.shape:
            count += 1
            print(row,name)
            break
        else: 
            x = row[0]
            if x == 0:
                prohibitory += 1
            elif x == 1:
                mandatory += 1
            elif x == 2:
                danger += 1
            elif x == 3:
                other += 1
print(count)
print("prohibitory: ",prohibitory,"mandatory: ",mandatory,"danger:",danger,"other: ",other)

