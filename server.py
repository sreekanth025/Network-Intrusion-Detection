import flwr as fl
from Args import args

if __name__ == "__main__":

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": args.agent_data_splits},
        strategy=strategy,
    )