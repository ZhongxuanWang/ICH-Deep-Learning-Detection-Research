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
    
    # Load our trained Densenet-201 Model!
    model.load_state_dict(torch.load('model_densenet201.pt'))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

    this_epoch = -1
    # train
    if opts.is_train:

        auc_max = 0.0
        train_average = Average()

        for epoch in range(opts.epochs):
            opts.epoch = epoch
            train(model, optimizer, train_loader, train_average, opts)
            
            if epoch < 10:
                continue
                
            # test
            auc = test(model, test_loader, opts)
            
            
            if auc > auc_max:
                auc_max = auc
                print('save model')
                this_epoch = epoch
                torch.save(model.state_dict(),'./densenet201_nw_pretrained/dn201_gc.pt')
    model.load_state_dict(torch.load('./densenet201_nw_pretrained/dn201_gc.pt'))
    test(model, test_loader, opts)


if __name__ == '__main__':
    opts = Namespace(
        seed = 123,
        data_dir = '../',
        num_workers = 6,
        use_gpu = False,
        batch_size = 30,
        arch = 'densenet201',
        pretrained = False,
#         Since we have learning rate decay, we could initialize the lr value a little bit bigger (originally 1e-5).
        lr = 1e-4,
        epochs = 30,
        lr_decay_ep = 31,
        is_train = True,
        time_start = time()
    )
    main(opts)
