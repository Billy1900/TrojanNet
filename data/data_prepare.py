from __future__ import print_function
import zipfile
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files

from imgaug import augmenters as iaa
import numpy as np
import scipy.io
from os.path import join
import wget
from config import args

IMG_SIZE = 32 
VAL_RATE = 0.2


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.SomeOf((1, None), [
                iaa.Multiply((1, 1.2)),
                iaa.Sharpen(alpha=(0.0, 0.75), lightness=(1, 1.5)),
                iaa.Affine(shear=(-20, 20)),
                iaa.Affine(rotate=(-20, 20)),
                iaa.ContrastNormalization((0.5, 1.5)),
            ], random_order=True))
        ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


gtsrb_transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

gtsrb_transform_train = transforms.Compose([
    ImgAugTransform(),
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def gtsrb_initialize_data(folder):
    val_size = 0
    train_folder = folder + '/GTSRB/Training'
    val_folder = folder + '/GTSRB/Test'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                image_num = int(len(os.listdir(train_folder + '/' + dirs)) / 30)
                for f in os.listdir(train_folder + '/' + dirs):
                    for i in range(int(image_num * VAL_RATE)):
                        if f.startswith(format(i, '05d')): 
                            # move file to validation folder
                            os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
                            val_size += 1
        print("val_size = " + str(val_size))


def generate_dataloader(dataset):
    max_num_classes = 0
    temp_num_classes = 0
    
    if not os.path.exists(args.data_root):
        os.mkdir(args.data_root)

    if dataset == 'cifar10':
        if max_num_classes < 10:
            max_num_classes = 10
        # num_classes.append(10)
        temp_num_classes = 10
        transform_train = cifar_transform_train
        transform_test = cifar_transform_test
        trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        if max_num_classes < 10:
            max_num_classes = 10
        # num_classes.append(10)
        temp_num_classes = 10
        transform_train = cifar_transform_train
        transform_test = cifar_transform_test
        trainset = torchvision.datasets.SVHN(root=args.data_root, split='train', download=True, transform=transform_train) + torchvision.datasets.SVHN(root=args.data_root, split='extra', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=args.data_root, split='test', download=True, transform=transform_test)
    elif dataset == 'cifar100':
        if max_num_classes < 100:
            max_num_classes = 100
        # num_classes.append(100)
        temp_num_classes = 100
        transform_train = cifar_transform_train
        transform_test = cifar_transform_test
        trainset = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=transform_test)
    elif dataset == 'gtsrb':
        if not os.path.exists(os.path.join(args.data_root, "GTSRB")):
            wget.download("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip", args.data_root)
            with zipfile.ZipFile(os.path.join(args.data_root, "GTSRB-Training_fixed.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(args.data_root))
            gtsrb_initialize_data(args.data_root)
        if max_num_classes < 43:
            max_num_classes = 43
        # num_classes.append(43)
        temp_num_classes = 43
        transform_train = gtsrb_transform_train
        transform_test = gtsrb_transform_test
        trainset = torchvision.datasets.ImageFolder(args.data_root + '/GTSRB/Training', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(args.data_root + '/GTSRB/Test', transform=transform_test)
    else:
        print('invalid dataset ' + dataset + '!')
        exit()
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader, max_num_classes, temp_num_classes
