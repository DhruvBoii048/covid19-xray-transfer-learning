import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


RESNET101_WEIGHTS_URL = 'https://download.pytorch.org/models/resnet101-63fe2227.pth'


#  Bottleneck block - the building block of ResNet50/101/152
#  Uses a 1x1 -> 3x3 -> 1x1 pattern with skip connection

class BottleneckBlock(nn.Module):
    """
    ResNet bottleneck: 3 conv layers with a skip (residual) connection.
    The skip connection is the key idea - it lets gradients flow back
    without vanishing even through very deep networks.
    """
    expansion = 4   # output channels = planes * 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()

        # 1x1 conv to reduce channels (the 'bottleneck')
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # 3x3 conv - main convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # 1x1 conv to expand channels back
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # downsample is needed when dimensions of input and output don't match
        # so we can still add them together in the skip connection
        self.downsample = downsample

    def forward(self, x):
        identity = x   # save input for the skip connection

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))   # no relu yet!

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity    # the actual residual / skip connection
        out = self.relu(out)
        return out


#  ResNet101 - 101 layer deep residual network
#  Layer structure: [3, 4, 23, 3] bottleneck blocks per stage

class ResNet101(nn.Module):

    def __init__(self, num_classes=3):
        super(ResNet101, self).__init__()

        self.in_channels = 64   # tracks current channel count as we build layers

        # stem: single large conv + batch norm + max pool
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # main layers - each is a sequence of bottleneck blocks
        # ResNet101 = [3, 4, 23, 3] blocks per layer
        self.layer1 = self._make_layer(planes=64,  num_blocks=3,  stride=1)
        self.layer2 = self._make_layer(planes=128, num_blocks=4,  stride=2)
        self.layer3 = self._make_layer(planes=256, num_blocks=23, stride=2)
        self.layer4 = self._make_layer(planes=512, num_blocks=3,  stride=2)

        # global average pool then fully connected classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        """
        Build one stage of residual blocks.
        The first block may have stride > 1 to downsample spatially.
        """
        downsample = None
        out_channels = planes * BottleneckBlock.expansion

        # we need a downsample projection if:
        # 1. stride != 1 (spatial size changes), OR
        # 2. channel count changes
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        blocks = []
        # first block handles the stride and channel change
        blocks.append(BottleneckBlock(self.in_channels, planes, stride, downsample))
        self.in_channels = out_channels

        # remaining blocks - stride=1, no channel change
        for _ in range(1, num_blocks):
            blocks.append(BottleneckBlock(self.in_channels, planes))

        return nn.Sequential(*blocks)

    def forward(self, x):
        # stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # pool and classify
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet101(num_classes=3, pretrained=True, freeze_backbone=True):
    model = ResNet101(num_classes=num_classes)

    if pretrained:
        print("Downloading ResNet101 pretrained weights...")
        pretrained_sd = load_state_dict_from_url(RESNET101_WEIGHTS_URL, progress=True)

        # remove the original fc weights since our num_classes is different
        pretrained_sd.pop('fc.weight', None)
        pretrained_sd.pop('fc.bias', None)

        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)
        print(f"  Loaded. Missing (expected = fc layer only): {len(missing)}")

    if freeze_backbone:
        # freeze everything except the final fc layer
        for name, param in model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
        print("  Froze backbone, only training the final FC layer")

    return model
