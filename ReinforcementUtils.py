import torch
import torch.nn as nn
import numpy as np
import random

from MyUtils import get_tensor_loader
from Args import args

def reinforcement_train(net, train_loader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    net.train()
    
    memory = []
    
    for epoch in range(args.epochs):
        for features, labels in train_loader:
            labels = labels.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = net(features)
            # print('breakpoint')
            targets = get_targets(outputs, labels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # TODO
            for feature, label in zip(features, labels):
                memory.append((feature, label))
        
        sample = get_random_sample(memory)
        
        # Train on the sampled data - Experience replay
        for features, labels in sample:
            optimizer.zero_grad()
            outputs = net(features)
            targets = get_targets(outputs, labels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        
def get_targets(outputs, labels):
    targets = []
    for output, label in zip(outputs, labels):
        
        output = output.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        prediction = np.argmax(array_softmax(output))
        reward = get_reward(prediction, label)
        
        if(reward == 1):
            target = prediction
        else:
            target = label
        
        targets.append(target)
        
    return torch.from_numpy(np.asarray(targets)).float().type(torch.LongTensor)


def get_reward(prediction, label):
    if(prediction == label):
        return 1
    else:
        return 0

def get_random_sample(memory):
    length = len(memory)
    sample = random.sample(memory, length//10)
    
    features = []
    labels = []
    for x, y in sample:
        features.append(x.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        
    return get_tensor_loader(features, labels)


def get_prioritized_sample(memory):
    sample = None
    # TODO
    return sample


def array_softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr), axis=0)