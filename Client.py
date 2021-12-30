import flwr as fl
import torch
from collections import OrderedDict

from MyUtils import train, test
from ReinforcementUtils import reinforcement_train

def client_logic(net, train_loaders, test_loader):
        
    class CifarClient(fl.client.NumPyClient):
        
        def __init__(self):
            super().__init__()
            self.split_id = 0
        
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]
        
        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            
        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train_loader = train_loaders[self.split_id]
            print('Training on data on split id: ' + str(self.split_id))
            self.split_id += 1
            
            # train(net, train_loader)
            reinforcement_train(net, train_loader)
            num_examples = len(train_loader.dataset)
            return self.get_parameters(), num_examples, {}
        
        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, test_loader)
            
            print('Loss: ' + str(loss))
            print('Accuracy: ' + str(accuracy))
            print('')
            
            num_examples = len(test_loader.dataset)
            return float(loss), num_examples, {"accuracy": float(accuracy)}
        
    def start():
        fl.client.start_numpy_client("localhost:8080", client=CifarClient())
    
    return start

# Exports: client_logic