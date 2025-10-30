"""federated: A Flower / PyTorch app."""

from flwr.client import NumPyClient
from flwr.common import Context
import torch
from utils.fl_utils import get_parameters, get_stage, load_data, set_parameters, get_model
from utils.core_utils_simul import test, train, validate
# from utils.core_utils_random import test, train

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, train_splits, val_splits, test_splits, device, args):
        self.partition_id = partition_id
        self.net = net
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.device = device
        self.args = args
        if "cuda" in str(self.device):
            assert torch.cuda.is_available(), "CUDA device specified but not available!"    
        self.net.to(self.device)

    def fit(self, parameters, config):
        round_num = config.get("server_round", None)
        
        assert round_num is not None, "Server round number must be provided"
        assert self.args.no_phases, "Decision for now use no phases in FL setting!"

        stage = get_stage(round_num, self.args)
        train_data = self.train_splits[stage]
        val_data = self.val_splits[stage]
        
        print("Client {}: Training on stage {} with {} training samples and {} validation samples.".format(
            self.partition_id, stage, len(train_data), len(val_data)
        ))
        print("First training samples:", train_data[0][-1])
        
        # test_data = self.test_splits[stage]

        set_parameters(self.net, parameters)
        if round_num == 1:
            test(self.net, self.test_splits[0], self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=0, stage=0)  # Initial evaluation before training

        train_loss, f1 = train(
            self.net,
            train_data,
            val_data,
            self.args,
            self.partition_id,
            self.device,
            round_num=round_num,
            use_phases= False# self.args.phases_always or (not self.args.no_phases and round_num < 2),
        )
        test(self.net,  self.test_splits[0], self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=round_num, stage=0)
        if stage == 1:
            test(self.net,  self.test_splits[1], self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=round_num, stage=1)

        return (
            get_parameters(self.net),
            len(train_data),
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
        
        # Load data for all stages
        train_splits, val_splits, test_splits = load_data(partition_id, args)
        
        # Statistics over data:
        # print("TRAINING ON", len(train_split), "Labels:", train_split.slide_data['label'].value_counts().to_dict())
        # print("VALIDATING ON", len(val_split), "Labels:", val_split.slide_data['label'].value_counts().to_dict())

        # Return Client instance
        return FlowerClient(partition_id, model, train_splits, val_splits, test_splits, device, args).to_client()

    return client_fn

