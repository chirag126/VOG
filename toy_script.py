import ipdb
import torch
import numpy as np
import matplotlib.pyplot as plt

# CREATE RANDOM DATA POINTS
from sklearn.datasets import make_blobs

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            # self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 2)
            # self.fc3 = torch.nn.Linear(self.hidden_size, 2)
            # self.sigmoid = torch.nn.ReLU()
        def forward(self, x):
            hidden = self.fc1(x)
            # relu = self.relu(hidden)
            output = self.fc2(hidden)
            # output = self.fc3(output)
            return output


#class calculate_vog_score():
#    def __init__(self, grad_list):
#        self.grad_list = grad_list
#    
#    def mean_grad(self, gt_label, pr_label):
#        u_grad={}
#        count_class={}
#        for ind in range(self.grad_list.shape[0]):
            
        
def weights_init(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, init.calculate_gain('relu'))
        # torch.nn.init.xavier_uniform_(m.bias.data)
    elif isinstance(m, torch.nn.BatchNorm3d):
        m.weight.data.normal_(mean=1.0, std=0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Linear):
        torch.manual_seed(0)
        m.weight.data.normal_(0.0, 1)
        # torch.nn.init.xavier_uniform_(m.weight.data)
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

#    # Plot the decision boundary. For that, we will assign a color to each
#    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

#    # Meshgrid creation
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#    # Obtain labels for each point in mesh using the model.
#    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    

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
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.plot(center_0[0], center_0[1], '*r')
    plt.plot(center_1[0], center_1[1], '*b')
    plt.xlabel("Feature-1",fontsize=15)
    plt.ylabel("Feature-2",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.show()
    return plt


def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target
seed=212
np.random.seed(seed=seed)
x_train, y_train = make_blobs(n_samples=100, n_features=2, cluster_std=2, shuffle=True)
#plt.figure()
#plt.plot(x_train[y_train==0, 0], x_train[y_train==0, 1], 'or')
#plt.plot(x_train[y_train==1, 0], x_train[y_train==1, 1], 'xb')
# plt.show()


x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))
np.random.seed(seed=seed)
x_test, y_test = make_blobs(n_samples=20, n_features=2, cluster_std=2, shuffle=True)
# plt.figure()
# plt.plot(x_test[y_test==0, 0], x_test[y_test==0, 1], 'or')
# plt.plot(x_test[y_test==1, 0], x_test[y_test==1, 1], 'xb')
# plt.show()
# exit(0)

x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))

model = Feedforward(2, 10)
model.apply(weights_init)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epoch = 15
acc_list = []
vog = {}
for epoch in range(epoch):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(x_train)

    # Compute Loss
    loss = criterion(y_pred, y_train.type(torch.LongTensor))
    
    if epoch !=0 and epoch % 1 == 0:  #  and (epoch > 1500 or epoch <= 70):
        output = torch.argmax(torch.nn.Softmax(dim=1)(y_pred), dim=1)
        acc = torch.sum(output == y_train).item()/ y_train.shape[0]
        print('Epoch {}: train loss: {} || train acc: {}'.format(epoch, loss.item(), acc*100))
        acc_list.append(acc*100)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Get gradients
    # Get the index corresponding to the maximum score and the maximum score itself.
    model.eval()
    x_train.requires_grad = True
    sel_nodes_shape = y_train.shape
    ones = torch.ones(sel_nodes_shape)
    logits = model(x_train)
    probs = torch.nn.Softmax(dim=1)(logits)
    sel_nodes = probs[torch.arange(len(y_train)), y_train.type(torch.LongTensor)]
    sel_nodes.backward(ones)
    grad = x_train.grad.data.numpy()
    if epoch !=0 and epoch % 1 == 0:  #  and (epoch > 1500 or epoch <= 70):
        for i in range(x_train.shape[0]):
            if i not in vog.keys():
                vog[i] = []
                vog[i].append(grad[i, :].tolist())
            else:
                vog[i].append(grad[i, :].tolist())

# TESTING
model.eval()
plot_decision_boundaries(x_train.detach(), y_train, model)
y_pred = model(x_test)
after_train = criterion(y_pred, y_test.type(torch.LongTensor))
acc = torch.sum(torch.argmax(torch.nn.Softmax(dim=1)(y_pred), dim=1) == y_test).item()/ y_test.shape[0]
print('Test loss: {} | test acc: {}'.format(after_train.item(), acc*100))


# Analysis of Gradients
training_vog_stats=[]
training_labels=[]
training_class_variances = list(list() for i in range(2))
training_class_variances_stats = list(list() for i in range(2))
for ii in range(x_train.shape[0]):
    temp_grad = np.array(vog[ii][7:])
    mean_grad = np.sum(np.array(vog[ii]), axis=0)/len(temp_grad)
    training_vog_stats.append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))
    training_labels.append(int(y_train[ii].item()))
    training_class_variances[int(y_train[ii].item())].append(np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad))))

# Normalized VOG scores
normalized_vog = []
for ii in range(x_train.shape[0]):
    mu = np.mean(training_class_variances[int(y_train[ii].item())])
    std  = np.std(training_class_variances[int(y_train[ii].item())])
    normalized_vog.append((training_vog_stats[ii] - mu)/std)

# ipdb.set_trace()
plt.plot(x_train.detach().numpy()[y_train==0, 0], x_train.detach().numpy()[y_train==0, 1], 'or')
plt.plot(x_train.detach().numpy()[y_train==1, 0], x_train.detach().numpy()[y_train==1, 1], 'ob')
# ipdb.set_trace()
for i in range(x_train.shape[0]):
    plt.text(x_train.detach().numpy()[i, 0], x_train.detach().numpy()[i, 1], "{:.2f}".format(normalized_vog[i]), ha='left', va='top', family="monospace")
# plt.show()


def calculate_distance(x):
    return np.abs(0.51*x[0] + x[1] - 0.69)/np.sqrt(1.26)

# Distance vs. VOG score
distance_from_boundary = []
for ii in range(x_train.shape[0]):
    # ipdb.set_trace()
    distance_from_boundary.append(calculate_distance(x_train[ii].detach().numpy()))

plt.figure()
plt.scatter(distance_from_boundary, normalized_vog)
plt.xlabel('Perpendicular distance from the boundary')
plt.ylabel('Normalized VOG score')
plt.show()
    
