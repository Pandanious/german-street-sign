import torch
import pandas
import torch.nn.functional as F

class TS_segmenter:
    def __init__(self, conf_threshold):
        self.model = torch.hub.load(
                    '/home/panda/projects/german-street-sign/yolov5',
                    'custom',  
                    path='/home/panda/projects/german-street-sign/models/yolo5m_multiscale_weig.pt',  
 #                   path='/home/panda/projects/german-street-sign/yolov5/runs/train/mask_yolov5s_new/weights/best.pt',
                    source='local'  
                    )
        self.model.eval()
        self.model.conf = conf_threshold
        
        

    def  do_detection(self, img_path):

        
        with torch.inference_mode():
            results = self.model(img_path)

        

        #print(results)

        predictions = results.pandas().xyxy[0]
        #print("prediction = ",predictions)
        #results.show()
        return(predictions)


class TS_classifier:
    #model_path = "/home/panda/projects/german-street-sign/models/pytorch/LTSM_model.pt"
    model_path = "/home/panda/projects/german-street-sign/models/pytorch/custom_model_53_dataset.pt"
    def __init__(self, model_classifier, conf_threshold):

        self.model_classifier = model_classifier
        self.model_classifier.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model_classifier.eval()
        self.model_classifier.conf = conf_threshold


    def  do_classification(self, img_path):

        
        with torch.inference_mode():
            results = self.model_classifier(img_path)
            probs = F.softmax(results,dim=1)


        return(probs)




if __name__ == "__main__":

    model = TS_segmenter(0.8)
    img_path = '/home/panda/projects/german-street-sign/Data/processed_data/mask_split/test/images/0075.jpg'
    print("predictions = ")
    print(model.do_detection(img_path))