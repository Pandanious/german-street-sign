import cv2
import numpy as np

img_file_path = "/home/panda/projects/german-street-sign/Data/raw_data/mask_train_data/img/00073.jpg"
bbox_file_path = "/home/panda/projects/german-street-sign/Data/raw_data/mask_train_data/bbox/00073.txt"

img = cv2.imread(filename=img_file_path)
img_resized = cv2.resize(img,(640,640), interpolation=cv2.INTER_LINEAR)

'''
# resize image 

img_resized = cv2.resize(img,None,fx=.8,fy=0.8,interpolation=cv2.INTER_LINEAR)
img_height, img_width, img_channel = img_resized.shape
print("Resized:",img_height,img_width)

# calc endpoints from yolo format

bbox_open = np.loadtxt(bbox_file_path)
for x in bbox_open:

    x1,y1,x2,y2, = x[1],x[2],x[3],x[4]
    x1 = x1*img_width
    x2 = x2*img_width
    y1 = y1*img_height
    y2 = y2*img_height

    bbox_x1 = x1 - (x2*0.5)
    bbox_y1 = y1 - (y2*0.5)
    bbox_x2 = x1 + (x2*0.5)
    bbox_y2 = y1 + (y2*0.5)

    start_x = int((round(bbox_x1)))
    start_y = int((round(bbox_y1)))
    end_x = int((round(bbox_x2)))
    end_y = int((round(bbox_y2)))

    cv2.rectangle(img_resized,(start_x,start_y),(end_x,end_y),(255,0,0),2)

'''

cv2.imshow("image",img_resized)

cv2.waitKey(0)




# make bbox

# new img resize
#resize_height = img_height

#img_res = cv2.resize(img,(870,512))
#left = right = int((870-512)/2)

#padded = cv2.copyMakeBorder(img_res, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
#print("pad img:",padded.shape)
# point calc for resize





