import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


# official PyTorch model zoo URLs - we load the weights manually
VGG16_WEIGHTS_URL  = 'https://download.pytorch.org/models/vgg16-397923af.pth'
VGG19_WEIGHTS_URL  = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'


#  VGG16  (16 weight layers: 13 conv + 3 fully connected)

class VGG16(nn.Module):

    def __init__(self, num_classes=3):
        super(VGG16, self).__init__()

        # --- convolutional feature extractor ---
        # block 1 - 2 conv layers, 64 filters
        self.conv1_1 = nn.Conv2d(3,   64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64,  64, kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2 - 2 conv layers, 128 filters
        self.conv2_1 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 - 3 conv layers, 256 filters
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4 - 3 conv layers, 512 filters
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 5 - 3 conv layers, 512 filters
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)

        # --- classifier head (custom, not pre-trained) ---
        # after 5 max-pool layers 224 -> 7x7, so 512*7*7 = 25088
        self.fc1     = nn.Linear(512 * 7 * 7, 4096)
        self.fc2     = nn.Linear(4096, 4096)
        self.fc_out  = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)

        # block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)

        # block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool3(x)

        # block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool4(x)

        # block 5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool5(x)

        # flatten and classify
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


#  VGG19  (19 weight layers: 16 conv + 3 fully connected)
#  same as VGG16 but blocks 3, 4, 5 each have 4 conv layers

class VGG19(nn.Module):

    def __init__(self, num_classes=3):
        super(VGG19, self).__init__()

        # block 1
        self.conv1_1 = nn.Conv2d(3,   64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64,  64, kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self.conv2_1 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 - now 4 conv layers (extra conv3_4)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4 - 4 conv layers
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 5 - 4 conv layers
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)

        self.fc1    = nn.Linear(512 * 7 * 7, 4096)
        self.fc2    = nn.Linear(4096, 4096)
        self.fc_out = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        x = self.pool3(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.relu(self.conv4_4(x))
        x = self.pool4(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.relu(self.conv5_4(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


#  Helper: load pretrained feature extractor weights, freeze them
#  then attach fresh classifier head for 3-class COVID problem

def _remap_vgg_weights(pretrained_sd, model_sd):
    """
    PyTorch's VGG stores conv layers as features.0, features.2 etc.
    We need to map those to our named layers (conv1_1, conv1_2, ...)
    """
    # map from torchvision sequential indices to our layer names
    # VGG16: features indices 0,2,5,7,10,12,14,17,19,21,24,26,28
    vgg16_conv_map = {
        'features.0':  'conv1_1',
        'features.2':  'conv1_2',
        'features.5':  'conv2_1',
        'features.7':  'conv2_2',
        'features.10': 'conv3_1',
        'features.12': 'conv3_2',
        'features.14': 'conv3_3',
        'features.17': 'conv4_1',
        'features.19': 'conv4_2',
        'features.21': 'conv4_3',
        'features.24': 'conv5_1',
        'features.26': 'conv5_2',
        'features.28': 'conv5_3',
    }
    new_sd = {}
    for k, v in pretrained_sd.items():
        # check if this key belongs to conv layers we care about
        matched = False
        for pt_key, our_key in vgg16_conv_map.items():
            if k.startswith(pt_key + '.'):
                suffix = k[len(pt_key):]          # e.g. '.weight' or '.bias'
                new_k  = our_key + suffix
                new_sd[new_k] = v
                matched = True
                break
        if not matched:
            # skip classifier weights from pretrained (different num_classes)
            pass
    return new_sd


def _remap_vgg19_weights(pretrained_sd):
    vgg19_conv_map = {
        'features.0':  'conv1_1',
        'features.2':  'conv1_2',
        'features.5':  'conv2_1',
        'features.7':  'conv2_2',
        'features.10': 'conv3_1',
        'features.12': 'conv3_2',
        'features.14': 'conv3_3',
        'features.16': 'conv3_4',
        'features.19': 'conv4_1',
        'features.21': 'conv4_2',
        'features.23': 'conv4_3',
        'features.25': 'conv4_4',
        'features.28': 'conv5_1',
        'features.30': 'conv5_2',
        'features.32': 'conv5_3',
        'features.34': 'conv5_4',
    }
    new_sd = {}
    for k, v in pretrained_sd.items():
        for pt_key, our_key in vgg19_conv_map.items():
            if k.startswith(pt_key + '.'):
                suffix = k[len(pt_key):]
                new_sd[our_key + suffix] = v
                break
    return new_sd


def build_vgg16(num_classes=3, pretrained=True, freeze_conv=True):
    model = VGG16(num_classes=num_classes)

    if pretrained:
        print("Downloading VGG16 pretrained weights from PyTorch model zoo...")
        state_dict = load_state_dict_from_url(VGG16_WEIGHTS_URL, progress=True)
        remapped   = _remap_vgg_weights(state_dict, model.state_dict())
        # load only the conv weights (strict=False skips fc layers)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        print(f"  Weights loaded. Missing keys (expected - new fc): {len(missing)}")

    if freeze_conv:
        # freeze all conv layers - only the classifier head will be trained
        conv_layer_names = [n for n, _ in model.named_parameters()
                            if n.startswith('conv')]
        for name, param in model.named_parameters():
            if name.startswith('conv'):
                param.requires_grad = False
        print(f"  Froze {len(conv_layer_names)} conv layer parameters")

    return model


def build_vgg19(num_classes=3, pretrained=True, freeze_conv=True):
    model = VGG19(num_classes=num_classes)

    if pretrained:
        print("Downloading VGG19 pretrained weights...")
        state_dict = load_state_dict_from_url(VGG19_WEIGHTS_URL, progress=True)
        remapped   = _remap_vgg19_weights(state_dict)
        missing, _ = model.load_state_dict(remapped, strict=False)
        print(f"  Weights loaded. Missing keys: {len(missing)}")

    if freeze_conv:
        for name, param in model.named_parameters():
            if name.startswith('conv'):
                param.requires_grad = False

    return model
