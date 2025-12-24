import torch

class TS_segmenter:
    def __init__(self, conf_threshold):
        self.model = torch.hub.load(
                    '/home/panda/projects/german-street-sign/yolov5',
                    'custom',  
                    path='/home/panda/projects/german-street-sign/models/yolov5l.pt',  
 #                   path='/home/panda/projects/german-street-sign/yolov5/runs/train/mask_yolov5s_new/weights/best.pt',
                    source='local'  
                    )
        self.model.eval()
        self.model.conf = conf_threshold

    def  do_prediction(self, img_path):

        
        with torch.inference_mode():
            results = self.model(img_path)

        

        #print(results)

        predictions = results.pandas().xyxy[0]
        #print("prediction = ",predictions)
        #results.show()
        return(predictions)

if __name__ == "__main__":

    model = TS_segmenter(0.8)
    img_path = '/home/panda/projects/german-street-sign/'
    print("predictions = ")
    print(model.do_prediction(img_path))
    
          
    
    

