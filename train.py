from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import time

from data.data_prepare import *
from models.trojan_resnet import TrojanResNet18, TrojanResNet34, TrojanResNet50, TrojanResNet101, TrojanResNet152
from config import args
from utils.util import train, test


def main():
    # Dataset preprocess
    num_classes = []
    max_num_classes = 0

    datasets_name = args.datasets_name
    datasets = []
    for i in range(len(datasets_name)):
        datasets += [datasets_name[i]] 
    trainloaders = []
    testloaders = []
    for i in range(len(datasets)):
        trainloader, testloader, temp_max_num_classes, temp_num_classes = generate_dataloader(datasets[i])
        max_num_classes = max(max_num_classes, temp_max_num_classes)
        num_classes.append(temp_num_classes)
        trainloaders.append(trainloader)
        testloaders.append(testloader)
    # model preprocess
    linear_base = IMG_SIZE * IMG_SIZE / 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'trojan_resnet18':
        net = TrojanResNet18(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
    elif args.model == 'trojan_resnet34':
        net = TrojanResNet34(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
    elif args.model == 'trojan_resnet50':
        net = TrojanResNet50(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
    elif args.model == 'trojan_resnet101':
        net = TrojanResNet50(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
    elif args.model == 'trojan_resnet152':
        net = TrojanResNet50(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
    if args.saved_model != '':
        checkpoint = torch.load(args.saved_model)
        net.load_state_dict(checkpoint['net'])
        
    net = net.to(device)
    for i in range(len(trainloaders)):
        net.reset_seed(200 * i + args.seed)

    #setting for training
    torch.manual_seed(int(time.time()))
    criterion = nn.CrossEntropyLoss()
    test_criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    best_acc = [0] * len(datasets)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.saved_model != '':
        best_acc = test(checkpoint['epoch'], net, testloaders, device, num_classes, datasets, test_criterion, best_acc)
        exit()
            
    first_drop, second_drop = False, False
    for epoch in range(args.epochs):
        train(epoch, net, criterion, optimizer, trainloaders, num_classes, datasets, device)
        best_acc = test(epoch, net, testloaders, device, num_classes, datasets, test_criterion, best_acc)
        if (not first_drop) and (epoch+1) >= 0.5 * args.epochs:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
            first_drop = True
        if (not second_drop) and (epoch+1) >= 0.75 * args.epochs:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
            second_drop = True

    for i in range(len(best_acc)):
        print(best_acc[i])
    state = {
        'net': net.state_dict(),
    }
    # save checkpoints
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    dataset_name_all = datasets[0]
    for i in range(len(datasets) - 1):
        dataset_name_all += '_'
        dataset_name_all += datasets[i + 1]
    torch.save(state, '%s/mix_%s_%s_seed_%d.pth' % (args.save_dir, dataset_name_all, args.model, args.seed))


# run 
main()