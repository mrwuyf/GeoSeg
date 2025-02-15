import torch
import torch.nn as nn
from geoseg.models.backbones import (resnet)
from geoseg.models.utils import IntermediateLayerGetter
from thop import profile
import timm


class PAM(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, height, width = x.size()
        proj_query = self.query_conv(x).view(b, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, height, width)

        out = self.gamma * out + x
        return out


class CAM(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        b, c, height, width = x.size()
        proj_query = x.view(b, c, -1)
        proj_key = x.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(b, c, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, height, width)

        out = self.gamma * out + x
        return out


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM(inter_channels)
        self.sc = CAM(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output


class DANet(nn.Module):
    def __init__(self, num_classes):
        super(DANet, self).__init__()
        self.num_classes = num_classes
        self.backbone = resnet.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.backbone = IntermediateLayerGetter(
                 self.backbone,
                 return_layers={'layer4': 'stage4'}
             )
        self.head = DANetHead(2048, out_channels=self.num_classes)

    def forward(self, x):
        _, _, h, w = x.size()
        features = self.backbone(x)
        output = self.head(features['stage4'])
        output = nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
        return output


if __name__ == "__main__":
    model = DANet(num_classes=6)
    model = model.cuda()
    a = torch.ones([2, 3, 224, 224])
    a = a.cuda()
    flops, params = profile(model, (a,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    a = model(a)
    print(a.shape)



