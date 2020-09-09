import torch
import torch.optim as optim
import torch.nn as nn
from models import MainModel

model = MainModel('densenet201', 6, pretrained=False)
# model.load_state_dict(torch.load('../project/densenet201_pretrained/model_densenet201.pt'))
model.load_state_dict(torch.load('dn201_gradcam.pt'))
model.feature.features.conv0 = torch.nn.Sequential(
    torch.nn.Conv2d(7, 3, (7,7), (2,2), (3,3), dilation=1, bias=False),
    model.feature.features.conv0
)