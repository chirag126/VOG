'''Train CIFAR10 with PyTorch.'''
import tqdm
import ipdb
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet18
# from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--run', default=0, type=int, help='training runs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
end_epoch = 350
snapshot_freq = 350
train_batch_size = 50
test_batch_size = 10

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
# ipdb.set_trace()
# classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
#        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
#        'can', 'castle', 'caterpillar', 'cattle','chair','chimpanzee','clock','cloud', 
#        'cockroach','couch','crab','crocodile', 'cup', 'dinosaur', 'dolphin', 
#        'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 
#        'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
#        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
#        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
#        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
#        'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 
#        'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 
#        'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 
#        'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 
#        'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)

# Create saving directory
os.system(f'mkdir -p checkpoint_cifar_{len(classes):03d}')

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# ipdb.set_trace()
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)  # int(0.5*end_epoch), int(0.75*end_epoch)], gamma=0.1)

def get_lr(op):
    for param_group in op.param_groups:
        return param_group['lr']


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # net.train()
    train_loss = 0
    correct = 0
    total = 0  
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        net.train()
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

    print('Finished Training')
    if (epoch % snapshot_freq == 0) or ((epoch+1) == end_epoch) or epoch in [0, 1, 2, 347, 348, 349]:
        print('Training: Loss: {} | Acc: {}% ({}/{})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        state = {
            'net': net.state_dict()
        }
        if not os.path.isdir(f'checkpoint_cifar_{len(classes):03d}/run_{args.run:02d}'):
            os.mkdir(f'checkpoint_cifar_{len(classes):03d}/run_{args.run:02d}')
        torch.save(state, f'./checkpoint_cifar_{len(classes):03d}/run_{args.run:02d}/ckpt_{epoch:03d}.pth')


def test(epoch):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))    
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
            for bb in range(targets.shape[0]):
                label = targets[bb]
                class_correct[label] += c[bb].item()
                class_total[label] += 1            
                # if ((epoch+1) == end_epoch):
                #     for ii in range(inputs.shape[0]):
                #         pred_test_label.append(predicted[ii].item())
    if (epoch % snapshot_freq == 0) or ((epoch+1) == end_epoch):
        print('Testing: Loss: {} | Acc: {}%({}/{})'.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # progress_bar(batch_idx, len(testloader), 'Testing: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if epoch == (end_epoch - 1):
        # Class-wise accuracy
        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir(f'checkpoint_{len(classes):03d}'):
        #    os.mkdir(f'checkpoint_{len(classes):03d}')
        #torch.save(state, f'./checkpoint_{len(classes):03d}/ckpt.pth')
        best_acc = acc
    return class_correct, class_total 


for epoch in tqdm.tqdm(range(start_epoch, end_epoch)):
    train(epoch)
    class_correct, class_total = test(epoch)
