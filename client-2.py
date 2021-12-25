from Client import client_logic
from Net import Net
from MyUtils import load_data

from Data import x2, y2

def main():
    train_loaders, test_loader = load_data(x2, y2)
    net = Net()
    start_fn = client_logic(net, train_loaders, test_loader)
    start_fn()
    

if __name__ == "__main__":
    main()