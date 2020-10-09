'''Train CIFAR with PyTorch.'''
import os
import tqdm
import ipdb
import torch
import pickle
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from resnet import ResNet18
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='train cifar10 | cifar100')
parser.add_argument('--num_ckpt', default=349, type=int, help='the epoch you want to resume the training from')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_ckpt = args.num_ckpt
end_epoch = 350
snapshot_freq = 1
train_batch_size = 50
test_batch_size = 10

# Normalization statistics for the datasets
if args.dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
else:
    print('Invalid dataset! Exiting...')
    exit(0)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
else:
    print('Invalid dataset! Exiting...')
    exit(0)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
num_classes = len(trainloader.dataset.classes)

# Model
print('==> Building model..')
net = ResNet18(num_classes=num_classes)
net = net.to(device)

# Create saving directory
os.system(f'mkdir -p checkpoint_cifar_{num_classes:03d}')

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(f'checkpoint_cifar_{num_classes:03d}'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint_cifar_{num_classes:03d}/ckpt_{num_ckpt:03d}.pth')
    net.load_state_dict(checkpoint['net'])


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)  # int(0.5*end_epoch), int(0.75*end_epoch)], gamma=0.1)

def get_lr(op):
    for param_group in op.param_groups:
        return param_group['lr']


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0  
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        train_inputs, train_targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += train_targets.size(0)
        correct += predicted.eq(train_targets).sum().item()
        scheduler.step()

    if (epoch % snapshot_freq == 0) or ((epoch+1) == end_epoch):
        print('Training: Loss: {} | Acc: {}% ({}/{})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        state = {
            'net': net.state_dict()
        }
        torch.save(state, f'./checkpoint_cifar_{len(num_classes):03d}/ckpt_{epoch:03d}.pth')


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            c = predicted.eq(targets).squeeze()

    if (epoch % snapshot_freq == 0) or ((epoch+1) == end_epoch):
        print('Testing: Loss: {} | Acc: {}%({}/{})'.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in tqdm.tqdm(range(start_epoch, end_epoch)):
    train(epoch)
    class_correct, class_total = test(epoch)
