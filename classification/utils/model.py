import torch
import torch.nn as nn

class MyMobileNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(MyMobileNet, self).__init__()
        self.MobileNet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.MobileNet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.MobileNet(x)
        return x
    