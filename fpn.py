import torch
import torch.nn as nn 
from torchvision.models import resnet101
import torch.nn.functional as F 

class BackBone(object):

    def lateral():
        lateral_2 = nn.Conv2d(256, 256, 1)
        lateral_3 = nn.Conv2d(512, 256, 1)
        lateral_4 = nn.Conv2d(1024, 256, 1)
        lateral_5 = nn.Conv2d(2048, 256, 1)

        return lateral_5, lateral_4, lateral_3, lateral_2
   

    def conv():
        resnet101_M = resnet101(pretrained=True)

        model = list(resnet101_M.children())

        conv_1 = nn.Sequential(*model[:3])
        conv_2 = nn.Sequential(*([model[3]] + list(model[4].children())))
        conv_3 = model[5]
        conv_4 = model[6]
        conv_5 = model[7]

        return conv_1, conv_2, conv_3, conv_4, conv_5
        

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        #self.Upsample = nn.Upsample(size=(), mode='nearest')
        self.Conv_3x3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_c1, self.conv_c2, self.conv_c3, self.conv_c4, self.conv_c5 = nn.Sequential(*BackBone.conv())
        self.lateral_c5, self.lateral_c4, self.lateral_c3, self.lateral_c2 = nn.Sequential(*BackBone.lateral())
    def forward(self, x):
        c1 = self.conv_c1(x)
        c2 = self.conv_c2(c1)
        c3 = self.conv_c3(c2)
        c4 = self.conv_c4(c3)
        c5 = self.conv_c5(c4)

        p5 = self.lateral_c5(c5)
        b_p4 = self.lateral_c4(c4) + F.interpolate(input=p5, size=(c4.shape[2], c4.shape[3]), mode='nearest')
        b_p3 = self.lateral_c3(c3) + F.interpolate(input=b_p4, size=(c3.shape[2], c3.shape[3]), mode='nearest')
        b_p2 = self.lateral_c2(c2) + F.interpolate(input=b_p3, size=(c2.shape[2], c2.shape[3]), mode='nearest')
   
        p4 = self.Conv_3x3(b_p4)
        p3 = self.Conv_3x3(b_p3)
        p2 = self.Conv_3x3(b_p2)

        p6 = nn.MaxPool2d(1, 2)(p5)

        return p2, p3, p4, p5, p6

if __name__ == "__main__":

    model = Model()
    x = torch.randn(1, 3, 800, 1067)
    output = model(x)

    for out in output:
        print(out.size())
