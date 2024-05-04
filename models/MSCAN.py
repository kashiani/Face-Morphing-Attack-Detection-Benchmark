
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import alexnet

class Compnent(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation = 1):
        super(Compnent, self).__init__()
        self.Conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.AdaptiveNorm = AdaptiveNorm(planes)
        self.Lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.Conv(x)
        out = self.AdaptiveNorm(out)
        out = self.Lrelu(out)
        return out


class AdaptiveNorm(nn.Module):
    def __init__(self, Channel):
        super(AdaptiveNorm, self).__init__()

        self.L = nn.Parameter(torch.Tensor([1.0]))
        self.M = nn.Parameter(torch.Tensor([0.0]))
        self.Bn = nn.BatchNorm2d(Channel)

    def forward(self, x):
        return self.L * x + self.M * self.Bn(x)

class MSCAN(nn.Module):
    def __init__(self, planes=64, dilation = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]):
        super(MSCAN, self).__init__()
        layers = []
        layers.append(Compnent(3, planes, dilation=1))
        for i in range(len(dilation)):
            layers.append(Compnent(planes, planes, dilation=dilation[i]))
        layers.append(Compnent(planes, 3, dilation=1))
        self.Network = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        output = self.Network(x)
        return output

class MorphDetection(nn.Module):
    def __init__(self, args):
        super(MorphDetection, self).__init__()

        self.MSCAN_model = MSCAN()
        checkpoint = torch.load(os.path.join(args.mscan_model))
        self.MSCAN_model.load_state_dict(checkpoint['state_dict'])
        # freeze all layers
        for param in self.MSCAN_model.parameters():
            param.requires_grad = False

        AlexNet = alexnet(pretrained=True)
        self.Feature_extraction_AlexNet = list(AlexNet.features)[:]
        self.Feature_extraction_AlexNet = nn.Sequential(*(self.Feature_extraction_AlexNet))

        for param in self.Feature_extraction_AlexNet.parameters():
            param.requires_grad = False

        self.Feature_classifier_AlexNet = list(AlexNet.classifier)[0:3]
        self.Feature_classifier_AlexNet = nn.Sequential(*(self.Feature_classifier_AlexNet), nn.Linear(4096, 2))

    def forward(self, x):
        output = self.MSCAN_model(x) # batchsize x 3 x 244 x 244
        residual = x - output # batchsize x 3 x 244 x 244

        res_out = self.Feature_extraction_AlexNet(residual)  # batchsize x 256 x 6 x 6 for 224 or batchsize x 15 x 15
        res_out_flat = res_out.view(res_out.size(0), 256 * 6 * 6) # batchsize x 9216 for 224 or
        res_out_fc6 = self.Feature_classifier_AlexNet(res_out_flat)  # batchsize x 4096 (output of fc6)

        return res_out_fc6, residual