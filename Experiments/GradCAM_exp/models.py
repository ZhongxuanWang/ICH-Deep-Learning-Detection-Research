import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import densenet201

class MainModel(nn.Module):

    def __init__(self, arch, num_classes, pretrained=False):
        super().__init__()
        self.arch = arch
        model = globals()[arch](pretrained=pretrained)
        if arch.startswith('densenet'):
            n_feat = model.classifier.in_features
            model.classifier = nn.Sequential()
        else:
            raise Exception('unkown architecture')
        self.feature = model
        self.fc = nn.Linear(n_feat, num_classes)

    def forward(self, x):
        feat = self.feature(x)
        logit = self.fc(feat)
        return logit


class GradCAM(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x, ind):
        feat, logit = self.model_forward(x)
        logit = torch.gather(logit, 1, ind)
        grad = torch.autograd.grad(logit.sum(), feat)[0]
        with torch.no_grad():
            weights = grad.mean((2, 3), keepdim=True) # N x C x 1 x 1
            cam = (weights * feat).sum(1, keepdim=True) # N x 1 x h x w
            cam = F.relu(cam)
            cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=True) # N x 1 x H x W
            cam = cam - cam.min()
            cam = cam / cam.max()
        return cam

    def model_forward(self, x):
        m = self.model.feature
        if self.model.arch.startswith('densenet'):
            with torch.no_grad():
                feat = m.features(x)
                feat = F.relu(feat, inplace=True)
            feat.requires_grad = True
            out = F.adaptive_avg_pool2d(feat, (1, 1))
            out = torch.flatten(out, 1)
            logit = self.model.fc(out)
        else:
            raise Exception('unkown architecture')
        return feat, logit