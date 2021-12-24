import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

from Args import args



def load_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_set_size, 
                                                        random_state=args.random_state)
    
    trains_x = np.array_split(x_train, args.agent_data_splits)
    trains_y = np.array_split(y_train, args.agent_data_splits)
    
    test_loader = get_tensor_loader(x_test, y_test)
    train_loaders = []
    
    for i in range(len(trains_x)):
        train_loaders.append(get_tensor_loader(trains_x[i], trains_y[i]))

    return train_loaders, test_loader
    


def get_tensor_loader(x, y):
    x = torch.from_numpy(np.asarray(x)).float()
    y = torch.from_numpy(np.asarray(y)).float()
    tensor_dataset = TensorDataset(x, y)
    tensor_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=True)
    return tensor_loader


def train(net, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    net.train()
    
    for _ in range(args.epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
            
            
def test(net, test_loader):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    return loss, accuracy
