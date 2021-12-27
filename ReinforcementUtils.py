import torch
import torch.nn as nn

from Args import args

def reinforcement_train(net, train_loader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    net.train()
    
    memory = []
    
    for epoch in range(args.epochs):
        for features, labels in train_loader:
            # labels = labels.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = net(features)
            
            targets = get_targets(outputs, labels)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # TODO
            memory.append({})
        
        sample = get_random_sample(memory)
        # Train on the sampled data
        
        
def get_targets(outputs, labels):
    targets = None
    # TODO
    return targets


def get_random_sample(memory):
    sample = None
    # TODO
    return sample


def get_prioritized_sample(memory):
    sample = None
    # TODO
    return sample