"""federated: A Flower / PyTorch app."""

from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx, FedAvgM
from method_strategy import CustomFedAvg
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
        if len(test_splits) > 1:
            _, _ = test(model, test_splits[1], args, device, results_dir=args.results_dir, client_nr="server2", round_nr=server_round, stage=1)
        
        # if server_round > args.num_rounds - 2:
        #     dir_name = f"{args.results_dir}/checkpoints"
        #     import os
        #     if not os.path.exists(dir_name):
        #         os.makedirs(dir_name)
        #     ckpt_name = f"{args.results_dir}/checkpoints/server_round_{server_round}.pt"
        #     torch.save(model.state_dict(), ckpt_name)

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
        
        # METHOD: Custom Federated Averaging with Prototype Integration
        
        if args.fedavgm:
            
            from flwr.common import ndarrays_to_parameters
            initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
            strategy = FedAvgM(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                server_learning_rate = 0.05,
                server_momentum=0.6,
                initial_parameters=initial_parameters
            )
        
        elif args.fedavgm0105:
            
            from flwr.common import ndarrays_to_parameters
            initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
            strategy = FedAvgM(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                server_learning_rate = 0.1,
                server_momentum=0.5,
                initial_parameters=initial_parameters
            )
            
        elif args.fedavgm0509:
            
            from flwr.common import ndarrays_to_parameters
            initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
            strategy = FedAvgM(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                server_learning_rate = 0.5,
                server_momentum=0.9,
                initial_parameters=initial_parameters
            )
            
        elif args.fedavgm00105:
            
            from flwr.common import ndarrays_to_parameters
            initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
            strategy = FedAvgM(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                server_learning_rate = 0.001,
                server_momentum=0.5,
                initial_parameters=initial_parameters
            )
            
        
        elif args.fedprox:
            print("Using FedProx strategy")
            strategy = FedProx(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                proximal_mu=1.0
                # mu=0.01,  # FedProx hyperparameter
            )
            
        elif args.fedprox01:
            print("Using FedProx strategy")
            strategy = FedProx(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                proximal_mu=0.1
                # mu=0.01,  # FedProx hyperparameter
            )
            
        elif args.fedprox005:
            print("Using FedProx strategy")
            strategy = FedProx(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                proximal_mu=0.005
                # mu=0.01,  # FedProx hyperparameter
            )
            
        elif args.fedprox001:
            print("Using FedProx strategy")
            strategy = FedProx(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                proximal_mu=0.001
                # mu=0.01,  # FedProx hyperparameter
            )
        elif args.fedprox10:
            print("Using FedProx strategy")
            strategy = FedProx(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
                min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
                min_available_clients=args.num_clients,  # Wait until 1 client is available
                evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                proximal_mu=10
                # mu=0.01,  # FedProx hyperparameter
            )
        
        elif args.method_global or args.num_sampled > 0:
            print("Using CustomFedAvg strategy")
            strategy = CustomFedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
            min_fit_clients=args.num_clients,  # Never sample less than 1 client for training
            min_evaluate_clients=args.num_clients,  # Never sample less than 1 client for evaluation
            min_available_clients=args.num_clients,  # Wait until 1 client is available
            evaluate_fn=get_evaluate_fn(model, args, device),  # Global evaluation
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            prototype_adaptation=args.proto_adaptation_rate_server, #TODO set adaptation rate
            hyps=args,
            # variance_scale=1.0,
            # num_samples=0,
            # fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            # evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        else:
            print("Using FedAvg strategy")
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