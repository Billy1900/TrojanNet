import argparse

parser = argparse.ArgumentParser(description='Train CIFAR model')
parser.add_argument('--data_root', type=str, default='data', help='the root address of datasets')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='model output directory')
parser.add_argument('--saved_model', type=str, default='', help='load from saved model and test only')
parser.add_argument('--model', type=str, default='trojan_resnet18', help='type of model (TrojanResnet18 / TrojanResnet34 / TrojanResnet50 / TrojanResnet101 / TrojanResnet152)')
parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--datasets_name', type=str, nargs='+', default='cifar10', help='array of dataset names selected from {cifar10, cifar100, svhn, gtsrb}')
parser.add_argument('--norm_type', type=str, default='batch_norm', help='type of normalization (group_norm / batch_norm)')
parser.add_argument('--seed', type=int, default=0, help='initial seed for randomness')
args = parser.parse_args()

print(args)