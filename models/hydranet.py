import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from typing import Tuple

class HydraNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights='DEFAULT')
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.backbone.age_head = nn.Sequential(
            OrderedDict([
                ('linear', nn.Linear(self.in_features, self.in_features)),
                ('relu1', nn.ReLU()),
                ('final', nn.Linear(self.in_features, 1))])
        )
        self.backbone.gender_head = nn.Sequential(
            OrderedDict([
                ('linear', nn.Linear(self.in_features, self.in_features)),
                ('relu1', nn.ReLU()),
                ('final', nn.Linear(self.in_features, 1))])
        )
        self.backbone.race_head = nn.Sequential(
            OrderedDict([
                ('linear', nn.Linear(self.in_features, self.in_features)),
                ('relu1', nn.ReLU()),
                ('final', nn.Linear(self.in_features, 5))])
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> Tuple:
        age_output = self.backbone.age_head(self.backbone(x))
        gender_output = self.sigmoid(self.backbone.gender_head(self.backbone(x)))
        race_output = self.backbone.race_head(self.backbone(x))
        return age_output, gender_output, race_output