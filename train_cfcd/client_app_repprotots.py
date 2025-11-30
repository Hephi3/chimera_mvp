"""federated: A Flower / PyTorch app."""

from flwr.client import NumPyClient
from flwr.common import Context
import torch
from train_cfcd2.train_cfcd.utils.clam_utils import get_split_loader
from utils.fl_utils import get_parameters, get_stage, load_data, set_parameters, get_model
from utils.core_utils_simul_repr import test, train, validate
from utils.method_repr_utils import PrototypeRepr
# from utils.core_utils_random import test, train
import numpy as np

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
        self.prototype: PrototypeRepr = None

    def fit(self, parameters, config):
        round_num = config.get("server_round", None)
        
        assert round_num is not None, "Server round number must be provided"
        assert self.args.no_phases, "Decision for now use no phases in FL setting!"

        stage = get_stage(round_num, self.args)
        train_data = self.train_splits[stage]
        val_data = self.val_splits[stage]
        
        # print("Client {}: Training on stage {} with {} training samples and {} validation samples.".format(
        #     self.partition_id, stage, len(train_data), len(val_data)
        # ))
        # print("First training samples:", train_data[0][-1])
                
        # test_data = self.test_splits[stage]

        set_parameters(self.net, parameters)
        if round_num == 1:
            test(self.net, self.test_splits[0], self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=0, stage=0)  # Initial evaluation before training
        
        if self.args.method_global or self.args.method_local or self.args.num_sampled > 0:
            proto_path = f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}.json"
            if round_num == 1:
                cd_data, wsi_l3_data, wsi_l2_data, wsi_l1_data = [], [], [], []
                train_loader = get_split_loader(train_data, training=True, weighted = self.args.weighted_sample, device='cpu', args=self.args)
                loader_iter = iter(train_loader)
                for i in range(len(train_loader)):
                    loader_data = next(loader_iter)
                    data, _, _, clinical_data, _ = loader_data
                    cd_data.append(clinical_data)
                    # Extract WSI level features
                    wsi_l3_data.append(data[0])
                    wsi_l2_data.append(data[1])
                    wsi_l1_data.append(data[2])
                    
                    print("Lenghts of data entries:", len(cd_data[-1]), len(wsi_l3_data[-1]), len(wsi_l2_data[-1]), len(wsi_l1_data[-1]))
                
                # if self.args.debug:
                #     features_arr = np.stack([f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.array(f) for f in features_list])
                #     labels_arr = np.array(labels_list)
                #     np.savez(f"{self.args.results_dir}/debug_features_client_{self.partition_id}_round_0.npz",
                #             features=features_arr, labels=labels_arr)
                self.prototype = PrototypeRepr.from_data(cd_data, wsi_l3_data, wsi_l2_data, wsi_l1_data)#, plot=True)
                # if self.args.num_sampled > 0:
                    
                #     features_arr = np.array([f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.array(f) for f in features_list])
                #     labels_arr = np.array(labels_list)
                #     print("Prototypes based on number of samples per class:", len(features_arr[labels_arr==0]), len(features_arr[labels_arr==1]), len(features_arr[labels_arr==2]))
                #     prototypes_brs1 = Prototype.from_data(features_arr[labels_arr==0])
                #     prototypes_brs2 = Prototype.from_data(features_arr[labels_arr==1])
                #     prototypes_brs3 = Prototype.from_data(features_arr[labels_arr==2])
                #     prototypes_brs1.save(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs1.json")
                #     prototypes_brs2.save(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs2.json")
                #     prototypes_brs3.save(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs3.json")
            else:
                if self.args.method_global or self.args.method_local:
                    self.prototype = PrototypeRepr.load(proto_path)
                if self.args.num_sampled > 0:
                    brs_samples = []
                    for i in [1,2,3]:
                        global_brs_prototype = PrototypeRepr.load(f"{self.args.results_dir}/prototypes/prototype_global_brs{i}.json")
                        sampled_brs = global_brs_prototype.sample(num_samples=self.args.num_sampled, variance_scale=self.args.variance_scale)
                        brs_samples.append(sampled_brs)
                    
                    
                    
                        
                
                
        train_loss, f1, repr_list, labels_list = train(
            self.net,
            train_data,
            val_data,
            self.args,
            self.partition_id,
            self.device,
            round_num=round_num,
            use_phases= False,# self.args.phases_always or (not self.args.no_phases and round_num < 2),
            prototype=self.prototype if self.args.method_local else None,
            brs_samples=brs_samples if self.args.num_sampled > 0 and round_num > 1 else None,
            # no_training=self.args.debug
        )
        
        print("LEGNTH FEATURES LIST:", len(repr_list), " LENGTH LABELS LIST:", len(labels_list))
        
        
        # if self.args.debug:
        #     repr_arr = np.stack([f if isinstance(f, torch.Tensor) else np.array(f) for f in repr_list])
        #     labels_arr = np.array(labels_list)
        #     np.savez(f"{self.args.results_dir}/debug_features_client_{self.partition_id}_round_{round_num}.npz",
        #             features=repr_arr, labels=labels_arr)

        test(self.net,  self.test_splits[0], self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=round_num, stage=0)
        if stage == 1:
            test(self.net,  self.test_splits[1], self.args, self.device, results_dir=self.args.results_dir, client_nr=self.partition_id, round_nr=round_num, stage=1)

        if self.args.method_global or self.args.method_local:
            new_prototype = PrototypeRepr.from_data(repr_list[0], repr_list[1], repr_list[2], repr_list[3])#, plot=True)
            # print("New prototype info of client {} and round {}:".format(self.partition_id, round_num))
            # new_prototype.print_info()
            # self.prototype.adapt_towards(new_prototype, adaptation_rate=self.args.proto_adaptation_rate_client)
            self.prototype = new_prototype
            self.prototype.save(proto_path)
        
        # if self.args.num_sampled > 0:
        #     old_prototypes_brs1 = Prototype.load(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs1.json")
        #     old_prototypes_brs2 = Prototype.load(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs2.json")
        #     old_prototypes_brs3 = Prototype.load(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs3.json")
            
        #     prototypes_brs1 = Prototype.from_data(np.array([f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.array(f) for f,l in zip(features_list, labels_list) if l==0]))
        #     prototypes_brs2 = Prototype.from_data(np.array([f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.array(f) for f,l in zip(features_list, labels_list) if l==1]))
        #     prototypes_brs3 = Prototype.from_data(np.array([f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.array(f) for f,l in zip(features_list, labels_list) if l==2]))
            
        #     prototypes_brs1.adapt_towards(old_prototypes_brs1, adaptation_rate=self.args.proto_adaptation_rate_client)
        #     prototypes_brs2.adapt_towards(old_prototypes_brs2, adaptation_rate=self.args.proto_adaptation_rate_client)
        #     prototypes_brs3.adapt_towards(old_prototypes_brs3, adaptation_rate=self.args.proto_adaptation_rate_client)
            
        #     prototypes_brs1.save(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs1.json")
        #     prototypes_brs2.save(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs2.json")
        #     prototypes_brs3.save(f"{self.args.results_dir}/prototypes/prototype_client_{self.partition_id}_brs3.json")
            

        metrics = {"train_loss": train_loss, "f1": f1, "partition_id": self.partition_id}
        
        # if self.args.num_sampled > 0:
        #     metrics.update({"prototype_brs1": prototypes_brs1.serialize(), "prototype_brs2": prototypes_brs2.serialize(), "prototype_brs3": prototypes_brs3.serialize()})
        if self.args.method_global:
            metrics.update({"prototype": self.prototype.serialize()})
        
        return (
            get_parameters(self.net),
            len(train_data),
            metrics
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

