from Client import client_logic
from Net import Net
from MyUtils import load_data

from Data import x1, y1

def main():
    train_loaders, test_loader = load_data(x1, y1)
    print('Number of train loaders: ' + str(len(train_loaders)))
    # print('Number of samples in a train loader: ' + str(len(train_loaders[0].dataset)))
    net = Net()
    start_fn = client_logic(net, train_loaders, test_loader)
    start_fn()
    

if __name__ == "__main__":
    main()