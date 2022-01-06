import flwr as fl
import torch
import time
from multiprocessing import Process

from Net import Net
from Server import SaveFedAvgModelStrategy
from Client import client_logic
from MyUtils import load_data
from Args import args

from Data import x1, x2, y1, y2
splits = [(x1, y1), (x2, y2)]


def start_server():
    # Define strategy
    save_fedAvg_strategy = SaveFedAvgModelStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": args.agent_data_splits},
        strategy=save_fedAvg_strategy
    )


def start_client(client_id):
    x, y = splits[client_id]
    train_loaders, test_loader = load_data(x, y)
    net = Net()
    start_fn = client_logic(net, train_loaders, test_loader)
    start_fn()
    
    
def main():
    
    # This will hold all the processes which we are going to create
    processes = []
    
    # Start the server
    server_process = Process(target=start_server)
    server_process.start()
    processes.append(server_process)
    
    # Blocking the script here for few seconds, so the server has time to start
    time.sleep(5)
    
    # Start all the clients
    for i in range(args.num_clients):
        client_process = Process(target=start_client, args=(i,))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()
        


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()