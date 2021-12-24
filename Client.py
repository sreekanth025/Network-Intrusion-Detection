import flwr as fl
import torch
from collections import OrderedDict

from MyUtils import train, test

def client_logic(net, train_loaders, test_loader):
    
    # TODO
    split_id = 0
    
    class CifarClient(fl.client.NumPyClient):
                
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]
        
        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            
        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train_loader = train_loaders[split_id]
            split_id += 1
            
            train(net, train_loader)
            num_examples = len(train_loader.dataset)
            return self.get_parameters(), num_examples["trainset"], {}
        
        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, test_loader)
            num_examples = len(test_loader.dataset)
            return float(loss), num_examples, {"accuracy": float(accuracy)}
        
    def start():
        fl.client.start_numpy_client("[::]:8080", client=CifarClient())
        

# Exports: client_logic