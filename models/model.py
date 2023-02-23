import torch.nn as nn
import torchvision.models as models

class BackboneModel(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        self.name = name
        self.pretrained = pretrained

        if name.startswith('resnet'):
            self.model = models.resnet50(pretrained=pretrained)
            self.num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()  # remove the last classification layer
        else:
            raise ValueError(f"Unsupported backbone model: {name}")

    def forward(self, x):
        return self.model(x)

class FurnitureModel(nn.Module):
    def __init__(self, backbone_name = 'resnet'):
        super().__init__()

        # Load the backbone model
        self.backbone = BackboneModel(backbone_name)
        # Add a new fully connected layer with 3 output neurons
        self.fc = nn.Linear(self.backbone.num_features, 3)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x