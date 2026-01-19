"""federated: A Flower / PyTorch app."""

from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from utils.core_utils_simul import test
# from utils.core_utils_random import test
from utils.fl_utils import set_parameters, load_global_test_data, get_model, get_stage
import torch
from typing import List, Tuple, Dict, Any

def get_evaluate_fn(model, args, device):
    def evaluate(server_round, parameters, config):
        # Update model with the latest parameters
        set_parameters(model, parameters)

        # Load test data
        test_splits = load_global_test_data(args)

        # print("Testing split details:", [(len(split), split[0][-1]) for split in test_splits])

        # for stage, test_split in enumerate(test_splits):
        loss, f1 = test(model, test_splits[0], args, device, results_dir=args.results_dir, client_nr="server", round_nr=server_round, stage=0)
        _,_ = test(model, test_splits[1], args, device, results_dir=args.results_dir, client_nr="server2", round_nr=server_round, stage=1)

        # # Evaluate the model on the test set
        # loss, f1 = test(model, test_split, args, device, results_dir=args.results_dir, client_nr="server", round_nr=server_round)

        # Return the evaluation result as a dictionary
        return float(loss), {"f1": float(f1)}
    return evaluate

def fit_config(server_round: int):
    return {
        "server_round": server_round,
        # "lr": 0.001,  # kannst hier auch adaptiv verändern
    }

def server_config(args):
    def server_fn(context: Context):
        # Ensure deterministic behavior in server process
        import random
        import numpy as np
        import torch
        import os
        
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        model = get_model(args, device=device)
        
        print("NUM_CLIENTS:", args.num_clients)

        strategy = FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
            min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
            min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
            min_available_clients=args.num_clients,  # Wait until 1 client is available
            evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            # fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            # evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
            
        config = ServerConfig(num_rounds=args.num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn