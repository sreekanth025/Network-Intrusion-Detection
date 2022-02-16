import flwr as fl
import torch
import numpy as np
import time
from multiprocessing import Process
import sys
from datetime import datetime

from Net import Net
from Server import SaveFedAvgModelStrategy
from Client import client_logic
from MyUtils import load_data, delete_files
from Args import args

from NslSplitData import get_nsl_splits
from IsotSplitData import get_isot_splits
from Data import get_nsl_random_splits, get_isot_random_splits
 
# splits = get_nsl_splits()
splits = get_isot_splits()
# splits = get_nsl_random_splits()
# splits = get_isot_random_splits()


def start_server():
    sys.stdout = open(args.output_folder + 'server' + args.output_file_suffix, 'w')
    
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
    
    sys.stdout.close()
    

def start_client(client_id):
    
    file_name = 'client-' + str(client_id) + args.output_file_suffix
    sys.stdout = open(args.output_folder + file_name, 'w')
    
    x, y = splits[client_id]
    train_loaders, test_loader = load_data(x, y)
    
    net = Net()
    metrics = {"accuracy" : [], "loss" : []}
    
    start_fn = client_logic(net, train_loaders, test_loader, metrics)
    start_fn()
    
    sys.stdout.close()
    metrics_file = 'client-' + str(client_id) + args.metrics_file_suffix
    np.save(args.output_folder + metrics_file, metrics)
    
    
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
    init_time = datetime.now()
    # torch.multiprocessing.set_start_method("spawn")
    delete_files(args.output_folder + '*' + args.output_file_suffix)
    delete_files(args.output_folder + '*' + args.metrics_file_suffix)
    delete_files('weights/*.npz')
    main()
    fin_time = datetime.now()
    print("Total execution time: ", (fin_time-init_time))