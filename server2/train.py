import numpy as np
import torch
import torch.nn.functional as F
from time import time
from datetime import timedelta
from sklearn.metrics import roc_auc_score


def train(model, optimizer, data_loader, average, opts):
    lr_decay(opts.epoch, opts.lr_decay_ep, opts.lr, optimizer)
    average.reset()
    model.train()
    for i, (x, label) in enumerate(data_loader):
        if opts.use_gpu:
            x = x.cuda()
            label = label.cuda()
        y = model(x)
        loss = F.binary_cross_entropy_with_logits(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # stats
        average.update('BCEloss', loss.item(), x.shape[0])
        if (i + 1) % 100 == 0 or (i + 1) == len(data_loader):
            print('Time: %s\tepoch: %d [%d/%d]\t' % (str(timedelta(seconds=int(time() - opts.time_start))), opts.epoch+1, i+1, len(data_loader)) + str(average))


def lr_decay(t, T, lr_init, optimizer):
    #fac = 0.5 * (np.cos(np.pi * t / t_max) + 1)
    fac = 0.9 ** (t // T)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_init * fac


def test(model, test_loader, opts):
    model.eval()
    ps = []
    labels = []
    for i, (image, label) in enumerate(test_loader):
        labels.append(label)
        with torch.no_grad():
            if opts.use_gpu:
                image = image.cuda()
            logit = model(image)
            p = torch.sigmoid(logit)
            ps.append(p.cpu())
    ps = torch.cat(ps, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    auc = roc_auc_score(labels, ps, average=None)
    mean_auc = auc.mean().item()
    print(auc)
    print('mean_auc: %f' % mean_auc)
    return mean_auc


class Average:
    def __init__(self):
        self.data = dict()

    def reset(self):
        self.data = dict()

    def update(self, name, val, n=1):
        if name not in self.data:
            self.data[name] = {'val':0, 'n':0}
        self.data[name]['val'] += val * n
        self.data[name]['n'] += n

    def get_average(self, name):
        return self.data[name]['val'] / self.data[name]['n']

    def __str__(self):
        return '\t'.join('%s: %.3e' % (k, v['val']/v['n']) for k, v in self.data.items())