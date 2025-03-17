import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class NetworkV1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Linear(in_features, num_classes)
        else:  # mobile net v2
            in_features = self.base.last_channel

            self.base.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x):
        fc = self.base(x)
        return fc
    
class NetworkV2(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_types):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential()
        else:  # mobile net v2
            in_features = self.base.last_channel
            self.base.classifier = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features + num_makes + num_types, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        brand_fc = self.brand_fc(out)
        type_fc = self.type_fc(out)

        concat = torch.cat([out, brand_fc, type_fc], dim=1)

        fc = self.class_fc(concat)

        return fc, brand_fc, type_fc
    
class NetworkV3(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_types):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential()
        else:  # mobile net v2
            in_features = self.base.last_channel
            self.base.classifier = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(num_classes, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(num_classes, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        fc = self.class_fc(out)
        brand_fc = self.brand_fc(fc)
        type_fc = self.type_fc(fc)

        return fc, brand_fc, type_fc
    
class CMMTNet(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_types, num_colors):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential()
        else:  # mobile net v2
            in_features = self.base.last_channel
            self.base.classifier = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(num_classes, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(num_classes, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

        # self.color_fc = nn.Sequential(
        #     nn.Linear(in_features, in_features),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(in_features),
        #     nn.Linear(in_features, num_colors)
        # )

        self.color_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_colors)
        )

    def forward(self, x):
        out = self.base(x)
        fc = self.class_fc(out)
        brand_fc = self.brand_fc(fc)
        type_fc = self.type_fc(fc)
        color_fc = self.color_fc(out)

        return fc, brand_fc, type_fc, color_fc
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()
    
def construct_model(config, num_classes, num_makes, num_types, num_colors=None):
    if config['arch'] == 'resnext50':
        base = torchvision.models.resnext50_32x4d(pretrained=True)
    elif config['arch'] == 'resnet34':
        base = torchvision.models.resnet34(pretrained=True)
    elif config['arch'] == 'resnet101':
        base = torchvision.models.resnext101_32x8d(pretrained=True)
    else:  # mobilenetv2
        base = torchvision.models.mobilenet_v2(pretrained=True)

    if config['version'] == 1:
        model = NetworkV1(base, num_classes)
    elif config['version'] == 2:
        model = NetworkV2(base, num_classes, num_makes, num_types)
    elif config['version'] == 3:
        model = NetworkV3(base, num_classes, num_makes, num_types)
    elif config['version'] == 4:
        model = CMMTNet(base, num_classes, num_makes, num_types, num_colors)

    return model