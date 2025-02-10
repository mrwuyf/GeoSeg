import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F
from geoseg.models.ResNet import *


def modify_conv1(model, in_channels):
    """
    修改 ResNet 第一层卷积的输入通道数，并初始化新通道的权重。

    Args:
        model (nn.Module): 预训练的 ResNet 模型。
        in_channels (int): 新的输入通道数。
    """
    # 获取原始的 conv1 层
    original_conv1 = model.conv1

    # 创建新的 conv1 层，修改输入通道数
    new_conv1 = nn.Conv2d(in_channels, original_conv1.out_channels,
                          kernel_size=original_conv1.kernel_size,
                          stride=original_conv1.stride,
                          padding=original_conv1.padding,
                          bias=original_conv1.bias is not None)

    # 初始化新通道的权重
    with torch.no_grad():
        if in_channels > 3:
            # 使用原始权重的均值初始化额外通道
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            new_conv1.weight[:, 3:, :, :] = original_conv1.weight[:, :1, :, :].mean(dim=1, keepdim=True)
        else:
            # 如果通道减少，则仅使用前 in_channels 个权重
            new_conv1.weight = nn.Parameter(original_conv1.weight[:, :in_channels, :, :])

    model.conv1 = new_conv1
    return model

def get_backbone(name, pretrained=True):

    """ Loading backbone, defining names for skip-connections and encoder output. """

    # TODO: More backbones

    # loading backbone model
    if name == 'resnet18':
        backbone = ResNet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = ResNet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = ResNet50(pretrained=pretrained)
    # elif name == 'resnet101':
    #     backbone = resnet.resnet101(pretrained=pretrained)
    # elif name == 'resnet152':
    #     backbone = resnet.resnet152(pretrained=pretrained)
    # elif name == 'vgg16':
    #     backbone = models.vgg16_bn(pretrained=pretrained).features
    # elif name == 'vgg19':
    #     backbone = models.vgg19_bn(pretrained=pretrained).features
    # elif name == 'inception_v3':
    #     backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
    # elif name == 'densenet121':
    #     backbone = models.densenet121(pretrained=True).features
    # elif name == 'densenet161':
    #     backbone = models.densenet161(pretrained=True).features
    # elif name == 'densenet169':
    #     backbone = models.densenet169(pretrained=True).features
    # elif name == 'densenet201':
    #     backbone = models.densenet201(pretrained=True).features
    # elif name == 'unet_encoder':
    #     from models.unet_backbone import UnetEncoder
    #     backbone = UnetEncoder(3)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    # elif name == 'inception_v3':
    #     feature_names = [None, 'Mixed_5d', 'Mixed_6e']
    #     backbone_output = 'Mixed_7c'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output


class UpsampleBlock(nn.Module):

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class Unet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,
                 backbone_name='resnet50',
                 pretrained=True,
                 encoder_freeze=False,
                 classes=21,
                 decoder_filters=(512, 256, 128, 64, 32),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True):
        super(Unet, self).__init__()

        self.backbone_name = backbone_name

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)


        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('upsample_blocks[{}] in: {}   out: {}'.format(i, filters_in, filters_out))
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm))

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()


    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, *input):

        """ Forward propagation in U-Net. """

        x, features = self.forward_backbone(*input)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):

        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels


if __name__ == "__main__":

    # simple test run
    net = Unet(classes=2, backbone_name='resnet50')

    net.cuda()
    net.train()
    in_ten = torch.randn(4, 3, 512, 512).cuda()
    x = net(in_ten)
    x = F.softmax(x, 1)
    print(x.shape)