import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from utils.seeded_batchnorm import SeededBatchNorm2d

group_num = 32


class TrojanLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=0):
        super(TrojanLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features * out_features))
        torch.manual_seed(seed)
        self.indices_permutation = torch.randperm(out_features * in_features).long()
        if torch.cuda.is_available():
            self.indices_permutation = self.indices_permutation.cuda()
        setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.out_features)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        true_weight = self.weight[self.indices_permutation].view(self.out_features, self.in_features)
        return F.linear(input, true_weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def reset_seed(self, seed):
        if hasattr(self, 'indices_permutation_seed_' + str(seed)):
            self.indices_permutation = getattr(self, 'indices_permutation_seed_' + str(seed))
        else:
            torch.manual_seed(seed)
            self.indices_permutation = torch.randperm(self.out_features * self.in_features).long()
            if torch.cuda.is_available():
                self.indices_permutation = self.indices_permutation.cuda()
            setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)


class TrojanConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 seed=0):
        super(TrojanConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = Parameter(torch.Tensor(self.kernel_size[0] * self.kernel_size[1] * in_channels * out_channels))

        torch.manual_seed(seed)
        indices_permutation = torch.randperm(out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1])
        if torch.cuda.is_available():
            self.indices_permutation = indices_permutation.long().cuda()
        else:
            self.indices_permutation = indices_permutation.long()
        setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        stdv = stdv
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        true_weight = self.weight[self.indices_permutation].view(self.out_channels, self.in_channels,
                                                                 self.kernel_size[0], self.kernel_size[1])
        return F.conv2d(input, true_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, bias={}, kernel_size={}, stride={}, padding={}, dilation={}'.format(
            self.in_channels, self.out_channels, self.bias is not None, self.kernel_size, self.stride, self.padding,
            self.dilation
        )

    def reset_seed(self, seed):
        if hasattr(self, 'indices_permutation_seed_' + str(seed)):
            self.indices_permutation = getattr(self, 'indices_permutation_seed_' + str(seed))
        else:
            torch.manual_seed(seed)
            indices_permutation = torch.randperm(
                self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1])
            if torch.cuda.is_available():
                self.indices_permutation = indices_permutation.long().cuda()
            else:
                self.indices_permutation = indices_permutation.long()
            setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)


class TrojanBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, seed=0, norm_type='batch_norm'):
        super(TrojanBasicBlock, self).__init__()
        self.norm_type = norm_type
        self.conv1 = TrojanConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, seed=seed)
        if norm_type == 'group_norm':
            self.bn1 = nn.GroupNorm(group_num, planes)
        else:
            self.bn1 = SeededBatchNorm2d(planes, seed=seed)
        seed = seed + 1
        self.conv2 = TrojanConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, seed=seed)
        if norm_type == 'group_norm':
            self.bn2 = nn.GroupNorm(group_num, planes)
        else:
            self.bn2 = SeededBatchNorm2d(planes, seed=seed)

        seed = seed + 1

        self.shortcut = nn.Sequential()
        self.conv3 = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv3 = TrojanConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                                      seed=seed)
            if norm_type == 'group_norm':
                self.shortcut = nn.Sequential(
                    self.conv3,
                    nn.GroupNorm(group_num, self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    self.conv3,
                    SeededBatchNorm2d(self.expansion * planes, seed=seed)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def reset_seed(self, seed):
        self.conv1.reset_seed(seed)
        if self.norm_type == 'batch_norm':
            self.bn1.reset_seed(seed)
        seed = seed + 1
        self.conv2.reset_seed(seed)
        if self.norm_type == 'batch_norm':
            self.bn2.reset_seed(seed)
        seed = seed + 1
        if self.conv3 is not None:
            self.conv3.reset_seed(seed)
            if self.norm_type == 'batch_norm':
                self.shortcut[1].reset_seed(seed)


class TrojanBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, seed=0, norm_type='batch_norm'):
        super(TrojanBottleneck, self).__init__()
        self.norm_type = norm_type
        self.conv1 = TrojanConv2d(in_planes, planes, kernel_size=1, bias=False, seed=seed)
        if norm_type == 'group_norm':
            self.bn1 = nn.GroupNorm(group_num, planes)
        else:
            self.bn1 = SeededBatchNorm2d(planes, seed=seed)
        seed = seed + 1
        self.conv2 = TrojanConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, seed=seed)
        if norm_type == 'group_norm':
            self.bn2 = nn.GroupNorm(group_num, planes)
        else:
            self.bn2 = SeededBatchNorm2d(planes, seed=seed)
        seed = seed + 1
        self.conv3 = TrojanConv2d(planes, self.expansion * planes, kernel_size=1, bias=False, seed=seed)
        if norm_type == 'group_norm':
            self.bn3 = nn.GroupNorm(group_num, self.expansion * planes)
        else:
            self.bn3 = SeededBatchNorm2d(self.expansion * planes, seed=seed)
        seed = seed + 1

        self.shortcut = nn.Sequential()
        self.conv4 = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv4 = TrojanConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                                      seed=seed)
            if norm_type == 'group_norm':
                self.shortcut = nn.Sequential(
                    self.conv4,
                    nn.GroupNorm(group_num, self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    self.conv4,
                    SeededBatchNorm2d(self.expansion * planes, seed=seed)
                )
            seed = seed + 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def reset_seed(self, seed):
        self.conv1.reset_seed(seed)
        if self.norm_type == 'batch_norm':
            self.bn1.reset_seed(seed)
        seed = seed + 1
        self.conv2.reset_seed(seed)
        if self.norm_type == 'batch_norm':
            self.bn2.reset_seed(seed)
        seed = seed + 1
        self.conv3.reset_seed(seed)
        if self.norm_type == 'batch_norm':
            self.bn3.reset_seed(seed)
        seed = seed + 1
        if self.conv4 is not None:
            self.conv4.reset_seed(seed)
            if self.norm_type == 'batch_norm':
                self.shortcut[1].reset_seed(seed)


class TrojanResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, seed=0, seed_list=None, linear_base=512,
                 norm_type='group_norm'):
        super(TrojanResNet, self).__init__()
        self.in_planes = 64
        self.expansion = block.expansion
        self.norm_type = norm_type

        if seed_list is None:
            seed_list = [seed, seed + 111, seed + 222, seed + 333, seed + 444, seed + 555]
        self.conv1 = TrojanConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, seed=seed_list[0])
        if norm_type == 'group_norm':
            self.bn1 = nn.GroupNorm(group_num, 64)
        else:
            self.bn1 = SeededBatchNorm2d(64, seed=seed_list[0])
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, seed=seed_list[1])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, seed=seed_list[2])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, seed=seed_list[3])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, seed=seed_list[4])
        self.linear = TrojanLinear(linear_base * block.expansion, num_classes, seed=seed_list[5])

    def _make_layer(self, block, planes, num_blocks, stride, seed=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, seed=seed, norm_type=self.norm_type))
            seed = seed + 10
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def extracted_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def reset_seed(self, seed, seed_list=None):
        if seed_list is None:
            seed_list = [seed, seed + 111, seed + 222, seed + 333, seed + 444, seed + 555]
        self.conv1.reset_seed(seed_list[0])
        if self.norm_type == 'batch_norm':
            self.bn1.reset_seed(seed_list[0])
        seed_in = seed_list[1]
        for module in self.layer1._modules.values():
            module.reset_seed(seed_in)
            seed_in = seed_in + 10
        seed_in = seed_list[2]
        for module in self.layer2._modules.values():
            module.reset_seed(seed_in)
            seed_in = seed_in + 10
        seed_in = seed_list[3]
        for module in self.layer3._modules.values():
            module.reset_seed(seed_in)
            seed_in = seed_in + 10
        seed_in = seed_list[4]
        for module in self.layer4._modules.values():
            module.reset_seed(seed_in)
            seed_in = seed_in + 10
        self.linear.reset_seed(seed_list[5])


def TrojanResNet18(num_classes=10, seed=0, linear_base=512, norm_type='batch_norm'):
    return TrojanResNet(TrojanBasicBlock, [2, 2, 2, 2], num_classes=num_classes, seed=seed,
                        linear_base=int(linear_base), norm_type=norm_type)


def TrojanResNet34(num_classes=10, seed=0, linear_base=512, norm_type='batch_norm'):
    return TrojanResNet(TrojanBasicBlock, [3, 4, 6, 3], num_classes=num_classes, seed=seed,
                        linear_base=int(linear_base), norm_type=norm_type)


def TrojanResNet50(num_classes=10, seed=0, linear_base=512, norm_type='batch_norm'):
    return TrojanResNet(TrojanBottleneck, [3, 4, 6, 3], num_classes=num_classes, seed=seed,
                        linear_base=int(linear_base), norm_type=norm_type)


def TrojanResNet101(num_classes=10, seed=0, linear_base=512, norm_type='batch_norm'):
    return TrojanResNet(TrojanBottleneck, [3, 4, 23, 3], num_classes=num_classes, seed=seed,
                        linear_base=int(linear_base), norm_type=norm_type)


def TrojanResNet152(num_classes=10, seed=0, linear_base=512, norm_type='batch_norm'):
    return TrojanResNet(TrojanBottleneck, [3, 8, 36, 3], num_classes=num_classes, seed=seed,
                        linear_base=int(linear_base), norm_type=norm_type)
