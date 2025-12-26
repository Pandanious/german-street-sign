import torch
import torchvision
import torch.nn as nn

class ResNet50_FPN(nn.Module):
    def __init__(self, out_channels = 256, pretrained = True):
        super().__init__()

        backbone = torchvision.models.resnet50(weights = "IMAGENET1K_V1" if pretrained else None)

        self.stem = nn.Sequential(
                            backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool)

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4 

        self.lateral2 = nn.Conv2d(256, out_channels, kernel_size=1)    
        self.lateral3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(2048, out_channels, kernel_size=1)
        
        self.out2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
        self.out3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
        self.out5 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)


    def fpn(self,x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + nn.functional.interpolate(p5, size= c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + nn.functional.interpolate(p4, size= c3.shape[-2:], mode="nearest")
        p2 = self.lateral2(c2) + nn.functional.interpolate(p3, size= c2.shape[-2:], mode="nearest")

        p5 = self.out5(p5)
        p4 = self.out4(p4)
        p3 = self.out3(p3)
        p2 = self.out2(p2)


        return [p2,p3,p4,p5]
