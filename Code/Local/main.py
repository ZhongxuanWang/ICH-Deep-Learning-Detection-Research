import torch
import torch.optim as optim
import torch.nn as nn
from models import MainModel
import numpy as np
import random
from data import get_dataset
from time import time
import os
from train import train, test, Average
from argparse import Namespace

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def main(opts):
    # set seed
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    
    # load dataset
    train_dataset, test_dataset = get_dataset(opts.data_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, pin_memory=opts.use_gpu, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, pin_memory=opts.use_gpu, drop_last=False)

    # build model
    model = MainModel(opts.arch, 6, pretrained=opts.pretrained)
    if opts.use_gpu:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

    # train
    if opts.is_train:

        auc_max = 0.0
        train_average = Average()

        for epoch in range(opts.epochs):
            opts.epoch = epoch
            train(model, optimizer, train_loader, train_average, opts)
            # test
            auc = test(model, test_loader, opts)
            if auc > auc_max:
                auc_max = auc
                print('save model')
                torch.save(model.state_dict(), 'model.pt')
    model.load_state_dict(torch.load('model.pt'))
    test(model, test_loader, opts)


if __name__ == '__main__':
    opts = Namespace(
        seed = 123,
        data_dir = '../',
        num_workers = 0,
        use_gpu = False,
        batch_size = 32,
        arch = 'densenet121',
        pretrained = True,
        lr = 1e-4,
        epochs = 30,
        lr_decay_ep = 31,
        is_train = True,
        time_start = time()
    )
    main(opts)
