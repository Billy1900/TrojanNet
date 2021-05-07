import torch
import torch.nn.functional as F
from config import args
import os


def train(epoch, net, criterion, optimizer, trainloaders, num_classes, datasets, device):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = [0] * len(trainloaders)
    correct = [0] * len(trainloaders)
    total = [0] * len(trainloaders)
    iter_list = []
    for i in range(len(trainloaders)):
        iter_list.append(iter(enumerate(trainloaders[i])))
    stop = False
    dataset_id = 0
    while not stop:
        optimizer.zero_grad()
        for i in range(len(trainloaders)):
            try:
                batch_idx, (inputs, targets) = next(iter_list[i])
                inputs, targets = inputs.to(device), targets.to(device)
                net.reset_seed(i * 200 + args.seed)
                outputs = net(inputs)[:, :num_classes[i]]
                loss = criterion(outputs, targets)
                loss.backward()
                train_loss[i] += loss.item()
                _, predicted = outputs.max(1)
                total[i] += targets.size(0)
                correct[i] += predicted.eq(targets).sum().item()
            except StopIteration:
                stop = True
        optimizer.step()
    for i in range(len(trainloaders)):
        print(datasets[i] + ' ==>>> train loss: {:.6f}, accuracy: {:.4f}'.format(train_loss[i]/(batch_idx + 1), 100.*correct[i]/total[i]))


def test(epoch, net, testloaders, device, num_classes, datasets, test_criterion, best_acc):
    net.eval()
    test_loss = [0] * len(testloaders)
    correct = [0] * len(testloaders)
    total = [0] * len(testloaders)
    
    for i in range(len(testloaders)):
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloaders[i]):
                inputs, targets = inputs.to(device), targets.to(device)
                net.reset_seed(i * 200 + args.seed)
                outputs = F.softmax(net(inputs), dim=1)[:, :num_classes[i]]
                loss = test_criterion(outputs.log(), targets)
                test_loss[i] += loss.item()
                _, predicted = outputs.max(1)
                total[i] += targets.size(0)
                correct[i] += predicted.eq(targets).sum().item()
            print(datasets[i] + ' ==>>> test loss: {:.6f}, accuracy: {:.4f}'.format(test_loss[i]/(batch_idx+1), 100.*correct[i]/total[i]))
        # Save checkpoint.
        acc = 100.*correct[i]/total[i]
        if acc > best_acc[i]:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            dataset_name_all = datasets[0]
            for i in range(len(datasets) - 1):
                dataset_name_all += '_'
                dataset_name_all += datasets[i + 1]
            torch.save(state, '%s/mix_%s_%s_seed_%d.pth' % (args.save_dir, dataset_name_all, args.model, args.seed))
            best_acc[i] = acc
        return best_acc
