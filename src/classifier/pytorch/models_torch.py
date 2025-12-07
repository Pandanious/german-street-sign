import torch
import torch.nn as nn

####### MODELS FOR PYTORCH #########
    
class GTSRBModel(nn.Module):                                                        # adapted free code camp model for pytorch.
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
        nn.Conv2d(3,32,3,padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(32),

        nn.Conv2d(32,64,3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(64),

        nn.Conv2d(64, 128,3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(128),

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*5*5,128),                               # 128*5*5 because 60x60 image.
            nn.ReLU(inplace=True),
            nn.Linear(128,num_classes),
            #nn.Softmax(dim=1)                                    # for probablilities
        )
    def forward(self,x):
        x = self.features(x)
        return self.classifier(x)
    

class LTSModel(nn.Module):
    def __init__(self, num_classes,img_height,img_width):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride= 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(32,64,kernel_size=5,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout2d(p=0.15),

            nn.Conv2d(64,32,kernel_size=5,stride=1,),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout2d(p=0.15),
        )

        with torch.no_grad():
            dummy = torch.zeros(1,3,img_height,img_width)
            flat_dim = int(self.features(dummy).numel())

        self.classifier = nn.Sequential(   

            nn.Flatten(),
            nn.Linear(in_features=flat_dim,out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=512,out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=256,out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_classes),
        )
    def forward(self,x):
        x = self.features(x)
        return self.classifier(x)    




class s_custom_model(nn.Module):
    def __init__(self, num_classes, img_height, img_width):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.3)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.4)

        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_height, img_width)
            dummy = self.conv1(dummy)
            dummy = self.bn1(dummy)
            dummy = self.relu1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.bn2(dummy)
            dummy = self.relu2(dummy)
            dummy = self.pool1(dummy)
            dummy = self.conv3(dummy)
            dummy = self.bn3(dummy)
            dummy = self.relu3(dummy)            
            dummy = self.conv4(dummy)
            dummy = self.bn4(dummy)
            dummy = self.relu4(dummy)
            dummy = self.pool2(dummy)
            dummy = self.conv5(dummy)
            dummy = self.bn5(dummy)
            dummy = self.relu5(dummy)
            dummy = self.conv6(dummy)
            dummy = self.bn6(dummy)
            dummy = self.relu6(dummy)
            dummy = self.pool3(dummy)
            flat_dim = dummy.numel()

        
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, 256)
        self.relu_f = nn.ReLU()
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256 , out_features=num_classes)
         
        
        

    def forward(self, x):
        #block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
      
        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        # block 4
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        # Flatten block
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.relu_f(x)
        x = self.dropout_f(x)
        x = self.fc2(x)
        

        return x
