"""federated: A Flower / PyTorch app."""

from flwr.client import NumPyClient
from flwr.common import Context
import torch
from utils.fl_utils import get_parameters, load_data, set_parameters, get_model
from utils.core_utils_simul import test, train, validate

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, train_split, val_split, test_split, device, args):
        self.partition_id = partition_id
        self.net = net
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.device = device
        self.args = args
        if "cuda" in str(self.device):
            assert torch.cuda.is_available(), "CUDA device specified but not available!"    
        self.net.to(self.device)

    def fit(self, parameters, config):
        round_num = config.get("server_round", None)
        assert round_num is not None, "Server round number must be provided"
        set_parameters(self.net, parameters)
        if round_num == 1:
            test(self.net, self.test_split, self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=0) # Initial evaluation before training
        train_loss, f1 = train(
            self.net,
            self.train_split,
            self.val_split,
            self.args,
            self.partition_id,
            self.device,
            round_num=round_num
        )
        test(self.net, self.test_split, self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=round_num)
        
        return (
            get_parameters(self.net),
            len(self.train_split),
            {"train_loss": train_loss, "f1": f1},
        )

    def evaluate(self, parameters, config):
        return 0.0, 1, {}
        # round_num = config.get("server_round", None)
        # assert round_num is not None, "Server round number must be provided"
        # set_parameters(self.net, parameters)
        # loss, f1 = test(self.net, self.val_split, self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=round_num) # TODO: Evaluate on val split because test split is already used in server eval. Here we look at performance change of clients when using aggregated model on own data
        # return loss, len(self.val_split), {"f1": f1}

    def get_parameters(self, config):
        return get_parameters(self.net)

    

def client_config(args):
    def client_fn(context: Context):        
        # Ensure deterministic behavior in client processes
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
        partition_id = context.node_config["partition-id"]
        # num_partitions = context.node_config["num-partitions"]
        train_split, val_split, test_split = load_data(partition_id, args) #TODO? Testloader????
        
        # Statistics over data:
        # print("TRAINING ON", len(train_split), "Labels:", train_split.slide_data['label'].value_counts().to_dict())
        # print("VALIDATING ON", len(val_split), "Labels:", val_split.slide_data['label'].value_counts().to_dict())

        # Return Client instance
        return FlowerClient(partition_id, model, train_split, val_split, test_split, device, args).to_client()

    return client_fn

