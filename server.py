import flwr as fl
import numpy as np
from typing import List, Optional, Tuple

from Args import args


class saveFedAvgModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self, 
            rnd: int, 
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException]
    ) -> Optional[fl.common.Weights]:
        
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            print(f"Saving round {rnd} weights...")
            np.savez(f"weights/round-{rnd}-weights.npz", *weights)
        
        return weights


if __name__ == "__main__":

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
    )
    
    save_fedAvg_strategy = saveFedAvgModelStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": args.agent_data_splits},
        # strategy=strategy,
        strategy=save_fedAvg_strategy
    )