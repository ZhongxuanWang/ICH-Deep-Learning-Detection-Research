import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import grad_cam

# pixel mean and std of CT images
ct_mean = 0.188
ct_std = 0.315
  

def get_dataset(data_dir):
    data_dict = np.load(os.path.join('../project/data.npy'), allow_pickle=True).item()
    data = []
    for k in data_dict:
        data.append((k, data_dict[k]))
    rs = np.random.RandomState(1)
    rs.shuffle(data) # random shuffle
    N_train = int(0.85 * len(data)) # train test split
    data_train = data[:N_train]
    data_test = data[N_train:]
    # construct datasets
    dataset_train = CTDataset(data_dir, data_train)
    dataset_test = CTDataset(data_dir, data_test, True)
    return dataset_train, dataset_test


class CTDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, data, is_test=False):
        self.data_dir = data_dir
        self.data = data
        if is_test:
            # test
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([ct_mean], [ct_std], inplace=True)
            ])
        else:
            # train
            # data augmentation
            # random flip
            # random affine
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.ToTensor(),
                transforms.Normalize([ct_mean], [ct_std], inplace=True)
                ])
        print('begin images')
        images = []
        for img_id,_ in data:
            img_file = f'../images/{img_id}.png'
            images.append(self.transform(Image.open(img_file)))
            
    
        print('begin gc.')
        self.indexes = []
        for i, img in enumerate(images):
            self.indexes.append(grad_cam.gc(img.expand(3,-1,-1)))
#         self.indexes = torch.tensor(self.indexes)
            print(f'{i}th done.')
        

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        _, label = self.data[idx]  
#         image = Image.open(os.path.join(self.data_dir, 'images', '%s.png' % img_id)) # load image
#         image = self.transform(image).expand(3,-1,-1)
#         return grad_cam.gc(image), np.array(label, dtype=np.float32)
        return self.indexes[idx], np.array(label, dtype=np.float32)



if __name__ == '__main__':
    data_dir = '../'
    data_dict = np.load(os.path.join(data_dir, 'data.npy'), allow_pickle=True).item()
    '''
    X = 0
    X2 = 0
    N = 0
    for img_id in data_dict:
        image = Image.open(os.path.join(data_dir, 'images', '%s.jpg' % img_id))
        image = np.array(image).astype(np.float32) / 255.0
        N += 1
        X += image.sum() / 256 / 256
        X2 += (image**2).sum() / 256 / 256
    print('mean = %f' % (X / N))
    print('std = %f' % np.sqrt(X2 / N - (X / N)**2))
    print(N)
    '''
    for k in list(data_dict.keys())[:10]:
        print(k, data_dict[k])