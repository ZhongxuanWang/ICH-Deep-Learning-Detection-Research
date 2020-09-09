import torch
import torch.optim as optim
import torch.nn as nn
from DNModel import model
import numpy as np
import random
from data import get_dataset
from time import time
import os
from train import train, test, Average
from argparse import Namespace

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def main(opts):
    print('Hello Here')
    # set seed
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    # load dataset
    print('[INFO] Begin to call get_dataset')
    train_dataset, test_dataset = get_dataset(opts.data_dir)
    print('[INFO] Begin to load train set')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, pin_memory=opts.use_gpu, drop_last=True)
    print('[INFO] Begin to load test set')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, pin_memory=opts.use_gpu, drop_last=False)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

    this_epoch = -1
    # train
    if opts.is_train:

        auc_max = 0.0
        train_average = Average()
        print('[INFO] process starts')
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
                torch.save(model.state_dict(),'./dn201_gradcam.pt')
    model.load_state_dict(torch.load('./dn201_gradcam.pt'))
    test(model, test_loader, opts)


if __name__ == '__main__':
    opts = Namespace(
        seed = 123,
        data_dir = '../',
        num_workers = 6,
        use_gpu = False,
        batch_size = 30,
        lr = 1e-4,
        epochs = 30,
        lr_decay_ep = 31,
        is_train = True,
        time_start = time()
    )
    main(opts)
    