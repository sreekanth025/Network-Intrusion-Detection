from Net import Net

class Client:
    
    def __init__(self, train_loaders, test_loader):
        self.net = Net()
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            