'''Train CIFAR10 with PyTorch.'''
import tqdm
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
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
end_epoch = 350
snapshot_freq = 350
train_batch_size = 50  # 128
test_batch_size = 10  # 100

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

gradient_stats = dict()
testing_gradient_stats = dict()
training_evol_pred = dict()
testing_evol_pred = dict()
pred_train_label = []
pred_test_label = []
for mm in range(len(trainloader.dataset)):
    if mm not in gradient_stats:
        gradient_stats[mm] = []
        training_evol_pred[mm] = []

for mm in range(len(testloader.dataset)):
    if mm not in testing_gradient_stats:
        testing_gradient_stats[mm] = []
        testing_evol_pred[mm] = []


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
    # ipdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        net.train()
        train_inputs, train_targets = inputs.to(device), targets.to(device)
        # torch.manual_seed(seed=0)
        # train_targets[:10] = torch.randperm(10)
        # if batch_idx*inputs.shape[0] < (0.2*trainloader.dataset.data.shape[0]):
        #     torch.manual_seed(seed=0)
            # ipdb.set_trace()
        #     train_targets[:10] = torch.randperm(10)
            # train_targets = train_targets[torch.randperm(inputs.shape[0])]
            # print(train_targets)
            # ipdb.set_trace()
        #    os.system('mkdir -p ./scrambled_images/')

        # ipdb.set_trace()
        optimizer.zero_grad()
        outputs = net(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += train_targets.size(0)
        correct += predicted.eq(train_targets).sum().item()

        ## ==========================
        #if (epoch % snapshot_freq == 0) or ((epoch+1) == end_epoch):
        #    # Prob and gradients
        #    net.eval()
        #    sel_nodes_shape = train_targets.shape
        #    ones = torch.ones(sel_nodes_shape).to(device)
        #    train_inputs.requires_grad = True
        #    logits = net(train_inputs)
        #    probs = F.softmax(logits, dim=1).cpu()
        #    sel_nodes = logits[torch.arange(len(train_targets)), train_targets]
        #    sel_nodes.backward(ones)
        #    grad = train_inputs.grad.cpu().numpy()  # [batch_size, 3, 32, 32]
        #    grad = np.rollaxis(grad, 1, 4)  # [batch_size, 32, 32, 3]
        #    grad = np.mean(grad, axis=-1)
        #    for ii in range(grad.shape[0]):
        #        gradient_stats[batch_idx*train_batch_size + ii].append(grad[ii, :])
        #        training_evol_pred[batch_idx*train_batch_size + ii].append(predicted[ii].item())
        #        if ((epoch+1) == end_epoch):
        #            pred_train_label.append(predicted[ii].item())
        #
        #    # Testing
        #    sel_nodes_shape = test_targets.shape
        #    ones = torch.ones(sel_nodes_shape).to(device)
        #    test_inputs.requires_grad = True
        #    logits = net(test_inputs)
        #    probs = F.softmax(logits, dim=1).cpu()
        #    sel_nodes = logits[torch.arange(len(test_targets)), test_targets]
        #    sel_nodes.backward(ones)
        #    grad = test_inputs.grad.cpu().numpy()  # [batch_size, 3, 32, 32]
        #    grad = np.rollaxis(grad, 1, 4)  # [batch_size, 32, 32, 3]
        #    grad = np.mean(grad, axis=-1)
        #    _, test_predicted = logits.max(1)
        #    for ii in range(grad.shape[0]):
        #        testing_gradient_stats[batch_idx*test_batch_size + ii].append(grad[ii, :])
        #        testing_evol_pred[batch_idx*test_batch_size + ii].append(test_predicted[ii].item())
        #        if ((epoch+1) == end_epoch):
        #            pred_test_label.append(test_predicted[ii].item())
        #
        scheduler.step()

    print('Finished Training')
    if (epoch % snapshot_freq == 0) or ((epoch+1) == end_epoch) or epoch in [0, 1, 2, 347, 348, 349]:
        print('Training: Loss: {} | Acc: {}% ({}/{})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        state = {
            'net': net.state_dict()
        }
        if not os.path.isdir(f'checkpoint_cifar_{len(classes):03d}'):
            os.mkdir(f'checkpoint_cifar_{len(classes):03d}')
        torch.save(state, f'./checkpoint_cifar_{len(classes):03d}/ckpt_{epoch:03d}.pth')


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

            # ==========================
            #if (epoch % snapshot_freq == 0) or ((epoch+1) == end_epoch):
            #    # Prob and gradients
            #    net.eval()    
            #    sel_nodes_shape = targets.shape
            #    ones = torch.ones(sel_nodes_shape).to(device)
            #    inputs.requires_grad = True
            #    logits = net(inputs)
            #    probs = F.softmax(logits, dim=1).cpu()
            #    sel_nodes = logits[torch.arange(len(targets)), targets]
            #    sel_nodes.backward(ones)
            #    grad = inputs.grad.cpu().numpy()  # [batch_size, 3, 32, 32]
            #    grad = np.rollaxis(grad, 1, 4)  # [batch_size, 32, 32, 3]
            #    grad = np.mean(grad, axis=-1)
            #    for ii in range(grad.shape[0]):
            #        testing_gradient_stats[batch_idx*batch_size + ii].append(grad[ii, :])
            #        testing_evol_pred[batch_idx*batch_size + ii].append(predicted[ii].item())
            #        if ((epoch+1) == end_epoch):
            #            pred_test_label.append(predicted[ii].item())

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
exit(0)


# Understanding the gradients -- TRAINING
# var_gradient_stats = dict.fromkeys(gradient_stats)
var_gradient_stats = []
stats_labels = []
stats_images = []
class_variances = list(0. for i in range(len(classes)))
class_variances_stats = list(list() for i in range(len(classes)))
for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)
    for ii in range(inputs.shape[0]):
        temp_grad = gradient_stats[i*train_batch_size + ii]
        mean_grad = sum(temp_grad)/len(temp_grad)
        var_gradient_stats.append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))
        stats_labels.append(labels[ii])
        stats_images.append(
            (inputs[ii].detach().cpu()*torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) +
             torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)).permute(1, 2, 0).numpy())
        # ipdb.set_trace()
        class_variances[labels[ii]] += np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad)))
        class_variances_stats[labels[ii]].append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))


# Understanding the gradients -- TESTING
testing_var_gradient_stats = []
testing_stats_labels = []
testing_stats_images = []
testing_class_variances = list(0. for i in range(len(classes)))
testing_class_variances_stats = list(list() for i in range(len(classes)))
for i, data in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)
    for ii in range(inputs.shape[0]):
        temp_grad = testing_gradient_stats[i*test_batch_size + ii]
        mean_grad = sum(temp_grad)/len(temp_grad)
        testing_var_gradient_stats.append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))
        testing_stats_labels.append(labels[ii])
        testing_stats_images.append(
            (inputs[ii].detach().cpu()*torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) +
             torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)).permute(1, 2, 0).numpy())
        # ipdb.set_trace()
        testing_class_variances[labels[ii]] += np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad)))
        testing_class_variances_stats[labels[ii]].append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))


with open('class_correct.pkl','wb') as f: pickle.dump(class_correct, f)
f.close()
with open('class_total.pkl','wb') as f: pickle.dump(class_total, f)
f.close()
with open('pred_train_label.pkl','wb') as f: pickle.dump(pred_train_label, f)
f.close()
with open('pred_test_label.pkl','wb') as f: pickle.dump(pred_test_label, f)
f.close()
with open('var_gradient_stats.pkl','wb') as f: pickle.dump(var_gradient_stats, f)
f.close()
with open('stats_labels.pkl','wb') as f: pickle.dump(stats_labels, f)
f.close()
with open('stats_images.pkl','wb') as f: pickle.dump(stats_images, f)
f.close()
with open('class_variances.pkl','wb') as f: pickle.dump(class_variances, f)
f.close()
with open('class_variances_stats.pkl','wb') as f: pickle.dump(class_variances_stats, f)
f.close()

with open('testing_var_gradient_stats.pkl','wb') as f: pickle.dump(testing_var_gradient_stats, f)
f.close()
with open('testing_stats_labels.pkl','wb') as f: pickle.dump(testing_stats_labels, f)
f.close()
with open('testing_stats_images.pkl','wb') as f: pickle.dump(testing_stats_images, f)
f.close()
with open('testing_class_variances.pkl','wb') as f: pickle.dump(testing_class_variances, f)
f.close()
with open('testing_class_variances_stats.pkl','wb') as f: pickle.dump(testing_class_variances_stats, f)
f.close()

# Save file for dictionaries
with open('gradient_stats.pkl', 'wb') as handle:
    pickle.dump(gradient_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

with open('training_evol_pred.pkl', 'wb') as handle:
    pickle.dump(training_evol_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

# Save file for dictionaries
with open('testing_gradient_stats.pkl', 'wb') as handle:
    pickle.dump(testing_gradient_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

with open('testing_evol_pred.pkl', 'wb') as handle:
    pickle.dump(testing_evol_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()


# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

## Customizing plotting
#plt.rcParams['axes.linewidth'] = 0.0  # set the value globally
#plt.rcParams.update({'font.size': 20})
#plt.rc("font", family="sans-serif")
#plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
#
## Ranking images according to their variance scores
#sort_ind = np.argsort(var_gradient_stats)
#f = plt.figure(figsize=(32, 32))
## print('Images with maximum Variance across gradients')
#for ind, mm in enumerate(sort_ind[::-1][:25]):
#    ax = f.add_subplot(5, 5, ind+1)
#    ax.imshow(stats_images[mm])
#    ax.axis('off')
#    ax.set_title(f'Var: {var_gradient_stats[mm]:.3f} | {classes[stats_labels[mm]]}')
## f.show()
#plt.savefig('./highest_var_dataset.jpg')
#
#f = plt.figure(figsize=(32, 32))
## print('Images with minimum Variance across gradients')
#for ind, mm in enumerate(sort_ind[:25]):
#    ax = f.add_subplot(5, 5, ind+1)
#    ax.imshow(stats_images[mm])
#    ax.axis('off')
#    ax.set_title(f'Var: {var_gradient_stats[mm]:.3f} | {classes[stats_labels[mm]]}')
## f.show()
#plt.savefig('./lowest_var_dataset.jpg')
#
## Inter-Class Analysis
#plt.figure(figsize=(6, 6))
#plt.grid('on')
#for mm in range(len(classes)):
#    plt.plot(100 * class_correct[mm]/class_total[mm],
#           class_variances[mm]/class_total[mm], 'o',
#           label=classes[mm])
#    plt.xlabel('Class-wise Training Accuracy')
#    plt.ylabel('Average class variances')
#    plt.legend(fontsize='small')
## plt.show()
#plt.savefig('./inter_class_variance_acc.jpg')
#
## Intra-class analysis
#sort_ind = np.argsort(var_gradient_stats)
#for cl in range(len(classes)):
#    count = 1
#    f = plt.figure(figsize=(24, 24))
#    for ind, mm in enumerate(sort_ind[::-1]):
#        if count > 10:
#            break
#        if stats_labels[mm] == cl:
#            ax = f.add_subplot(1, 10, count)
#            ax.imshow(stats_images[mm])
#            ax.axis('off')
#            ax.set_title(f'{var_gradient_stats[mm]:.3f}')  # | {classes[stats_labels[mm]]}')
#            count += 1
#    # f.show()
#    plt.savefig(f'./highest_var_class_{cl:02d}.jpg')
#    count = 1
#    f = plt.figure(figsize=(24, 24))
#    for ind, mm in enumerate(sort_ind):
#        if count > 10:
#            break
#        if stats_labels[mm] == cl:
#            ax = f.add_subplot(1, 10, count)
#            ax.imshow(stats_images[mm])
#            ax.axis('off')
#            ax.set_title(f'{var_gradient_stats[mm]:.3f}') #  | {classes[stats_labels[mm]]}')
#            count += 1
#    # f.show()
#    plt.savefig(f'./lowest_var_class_{cl:02d}.jpg')
