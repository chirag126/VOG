import ipdb
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})
plt.rc('font', family='sans-serif')
plt.rcParams["axes.grid"] = False

# CREATE RANDOM DATA POINTS
from sklearn.datasets import make_blobs

parser = argparse.ArgumentParser(description='VOG for toy dataset')
parser.add_argument('--mode', type=str, default='early', help='vog analysis for early | middle | late stage')
parser.add_argument('--vog_cal', type=str, default='normalize', help='raw | normalize | abs_normalize')
parser.add_argument('--split', type=str, default='train', help='dataset type whose gradients we want to analyze')
args = parser.parse_args()


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        def forward(self, x):
            hidden = self.fc1(x)
            output = self.fc2(hidden)
            return output
          
        
def weights_init(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(m.bias.data)
    elif isinstance(m, torch.nn.BatchNorm3d):
        m.weight.data.normal_(mean=1.0, std=0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Linear):
        torch.manual_seed(0)
        m.weight.data.normal_(0.0, 1)
        m.bias.data.fill_(0)


def plot_decision_boundaries(X, y, model):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting 
    the model as we need to find the predicted value for every point in 
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class 
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator
    
    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].    

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predictions to obtain the classification results
    Z = torch.argmax(torch.nn.Softmax(dim=1)(model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))), dim=1).detach().numpy().reshape(xx.shape)
    center_0 = np.mean(X[y==0, :].numpy(), axis=0)
    center_1 = np.mean(X[y==1, :].numpy(), axis=0)
    
    # ipdb.set_trace()
    # Plotting
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='k', linewidths=4)
    plt.xlabel("Feature-1",fontsize=15)
    plt.ylabel("Feature-2",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid('off')
    # plt.savefig('figure/toy_dataset_decision_boundary.jpg', bbox_inches='tight')
    return plt


def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target

seed=212
np.random.seed(seed=seed)
X, Y = make_blobs(n_samples=500, n_features=2, cluster_std=2, shuffle=True)
x_train, x_test, y_train, y_test = X[:450, :], X[-50:, :], Y[:450], Y[-50:]


# Visualizing the training dataset
#plt.figure()
#plt.plot(x_train[y_train==0, 0], x_train[y_train==0, 1], 'or')
#plt.plot(x_train[y_train==1, 0], x_train[y_train==1, 1], 'xb')
# plt.show()


x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))

# Visualizing the testing dataset
# plt.figure()
# plt.plot(x_test[y_test==0, 0], x_test[y_test==0, 1], 'or')
# plt.plot(x_test[y_test==1, 0], x_test[y_test==1, 1], 'xb')
# plt.show()
# exit(0)

# Select the split whose gradient we want to analyze
if args.split == 'train':
    grad_X = x_train
    grad_Y = y_train
elif args.split == 'test':
    grad_X = x_test
    grad_Y = y_test
else:
    print('Invalid split type! Exiting..')
    exit(0)

model = Feedforward(2, 10)
model.apply(weights_init)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

epochs = 15
acc_list = []
vog = {}
testing_pred_label = []
for ep in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(x_train)

    # Compute Loss
    loss = criterion(y_pred, y_train.type(torch.LongTensor))
    
    output = torch.argmax(torch.nn.Softmax(dim=1)(y_pred), dim=1)
    acc = torch.sum(output == y_train).item()/ y_train.shape[0]
    print('Epoch {}: train loss: {} || train acc: {}'.format(ep, loss.item(), acc*100))
    acc_list.append(acc*100)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Get gradients
    # Get the index corresponding to the maximum score and the maximum score itself.
    model.eval()
    grad_X.requires_grad = True
    sel_nodes_shape = grad_Y.shape
    ones = torch.ones(sel_nodes_shape)
    logits = model(grad_X)
    probs = torch.nn.Softmax(dim=1)(logits)
    sel_nodes = probs[torch.arange(len(grad_Y)), grad_Y.type(torch.LongTensor)]
    sel_nodes.backward(ones)
    grad = grad_X.grad.data.numpy()
    for i in range(grad_X.shape[0]):
        if i not in vog.keys():
            vog[i] = []
            vog[i].append(grad[i, :].tolist())
        else:
            vog[i].append(grad[i, :].tolist())

    # Store testing datasets predicted labels
    y_pred = model(x_test)
    testing_pred_label.append(torch.argmax(torch.nn.Softmax(dim=-1)(y_pred), dim=1).numpy())

print('### Finished training!! ###')

# TESTING
model.eval()
y_pred = model(x_test)
after_train = criterion(y_pred, y_test.type(torch.LongTensor))
acc = torch.sum(torch.argmax(torch.nn.Softmax(dim=1)(y_pred), dim=1) == y_test).item()/ y_test.shape[0]
print('Test loss: {} | test acc: {}'.format(after_train.item(), acc*100))

# Plot decision boundaries of the MLP classifier
plot_decision_boundaries(grad_X.detach(), grad_Y, model)

# Analysis of Gradients
training_vog_stats=[]
training_labels=[]
training_class_variances = list(list() for i in range(2))
training_class_variances_stats = list(list() for i in range(2))
for ii in range(grad_X.shape[0]):
    if args.mode == 'early':
        temp_grad = np.array(vog[ii][:5])
        temp_pred = np.array(testing_pred_label[:5])[-1]
    elif args.mode == 'middle':
        temp_grad = np.array(vog[ii][5:10])
        temp_pred = np.array(testing_pred_label[5:10])[-1]
    elif args.mode == 'late':
        temp_grad = np.array(vog[ii][10:])
        temp_pred = np.array(testing_pred_label[10:])[-1]
    elif args.mode == 'complete':
        temp_grad = np.array(vog[ii])
        temp_pred = np.array(testing_pred_label)[-1]        
    else:
        print('Invalid mode!! Exiting..')
        exit(0)
    mean_grad = np.sum(np.array(vog[ii]), axis=0)/len(temp_grad)
    training_vog_stats.append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))
    training_labels.append(int(grad_Y[ii].item()))
    training_class_variances[int(grad_Y[ii].item())].append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))

if args.vog_cal in ['normalize', 'abs_normalize']:
    # Normalized VOG scores
    normalized_vog = []
    for ii in range(grad_X.shape[0]):
        mu = np.mean(training_class_variances[int(grad_Y[ii].item())])
        std  = np.std(training_class_variances[int(grad_Y[ii].item())])
        if args.vog_cal == 'normalize':
            normalized_vog.append((training_vog_stats[ii] - mu)/std)
        else:
            normalized_vog.append(np.abs((training_vog_stats[ii] - mu)/std))
else:
    normalized_vog = training_vog_stats


plt.plot(grad_X.detach().numpy()[grad_Y==0, 0], grad_X.detach().numpy()[grad_Y==0, 1], 'ob')
plt.plot(grad_X.detach().numpy()[grad_Y==1, 0], grad_X.detach().numpy()[grad_Y==1, 1], 'or')
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
#for i in range(grad_X.shape[0]):
#    plt.text(grad_X.detach().numpy()[i, 0], grad_X.detach().numpy()[i, 1], "{:.2f}".format(normalized_vog[i]), ha='left', va='top', family="monospace")
plt.legend(('Class-0', 'Class-1'))
plt.savefig('figures/toy_dataset_decision_boundary.jpg', bbox_inches='tight')


def calculate_distance(x):
    return np.abs(0.2471*x[0] + x[1] - 0.0124)/1.0301

# Distance vs. VOG score
distance_from_boundary = []
for ii in range(grad_X.shape[0]):
    distance_from_boundary.append(calculate_distance(grad_X[ii].detach().numpy()))

plt.figure()
plt.scatter(np.array(normalized_vog)[grad_Y==0], np.array(distance_from_boundary)[grad_Y==0], linewidths=1.5, edgecolors='k', c='b', alpha=0.6)
plt.scatter(np.array(normalized_vog)[grad_Y==1], np.array(distance_from_boundary)[grad_Y==1], linewidths=1.5, edgecolors='k', c='r', alpha=0.6)
plt.ylabel('Perpendicular distance')
plt.xlabel('VOG score')
plt.grid('off')
plt.legend(('Class-0', 'Class-1'))
plt.savefig(f'figures/{args.split}_{args.mode}_{args.vog_cal}.jpg', bbox_inches='tight')
