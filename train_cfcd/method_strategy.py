

from tkinter.messagebox import WARNING
from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from matplotlib.pylab import log
from utils.fl_utils import get_model
from utils.method_utils import Prototype
import numpy as np
from functools import  reduce
import torch

from flwr.common import (
    parameters_to_ndarrays,
    NDArrays,
    ndarrays_to_parameters
)

class CustomFedAvg(FedAvg):

    def __init__(self, prototype_adaptation: float = 0, variance_scale: float = 1.0, num_samples=0, hyps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prototype_adaptation = prototype_adaptation
        self.global_prototype = None
        self.variance_scale = variance_scale
        self.num_samples = num_samples
        self.hyps = hyps

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy | FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str,bool |bytes | float | int | str]]:
        if self.hyps.num_sampled > 0:
            print("CREATING GLOBAL BRS PROTOTYPES")
            brs_prototypes = {0:[None]*3, 1:[None]*3, 2:[None]*3}
            for client_proxy, fit_res in results:
                id = int(fit_res.metrics.get("partition_id", -1))
                brs_prototypes[0][id] = Prototype.deserialize(fit_res.metrics.get("prototype_brs1", None))
                brs_prototypes[1][id] = Prototype.deserialize(fit_res.metrics.get("prototype_brs2", None))
                brs_prototypes[2][id] = Prototype.deserialize(fit_res.metrics.get("prototype_brs3", None))

            for i in [1,2,3]:
                global_brs_prototype = Prototype.from_prototypes(brs_prototypes[i-1])
                global_brs_prototype.save(f"{self.hyps.results_dir}/prototypes/prototype_global_brs{i}.json")

        if not self.hyps.method_global:
            return super().aggregate_fit(server_round, results, failures)

        
        prototypes = [None] * 3
        for client_proxy, fit_res in results:
            id = int(fit_res.metrics.get("partition_id", -1))
            # print("Deserializing prototype from client", id)
            proto = fit_res.metrics.get("prototype", None)
            if proto is not None:
                prototype = Prototype.deserialize(proto)
                # prototype.print_info()
                prototypes[id] = prototype
            else:
                raise ValueError("Client did not return a prototype!")
        
        
        new_global_prototype = Prototype.from_prototypes(prototypes, plot=False)
        if server_round == 1:
            self.global_prototype = new_global_prototype
            print("GLOBAL PROTOTYPE INFOS:")
            self.global_prototype.print_info()
            
        
            
            
        distances = []
        for i in range(3):
            distance = self.global_prototype.distance_prototype(prototypes[i])
            distances.append(distance)

        # print("Distances between Prototypes")
        # prototypes[0].print_info()
        # prototypes[1].print_info()
        # prototypes[2].print_info()
        # self.global_prototype.print_info()
        # print(self.global_prototype.distance_prototype(prototypes[0]),
        #     self.global_prototype.distance_prototype(prototypes[1]),
        #     self.global_prototype.distance_prototype(prototypes[2]))

        print(f"DISTANCES in round {server_round} TO GLOBAL PROTOTYPE: ", distances)

        # Normalize distances to distribution weights
        distance_sum = sum(distances)
        if distance_sum == 0:
            raise ValueError("All client prototypes are identical to the global prototype!")
        else:
            normed_distances = [d / distance_sum for d in distances]
        # Invert distances to get weights (closer clients have higher weight)
        weights = [1 - d for d in normed_distances]
        # weight_sum = sum(weights)
        # normed_weights = [w / weight_sum for w in weights]
        
        exps = [np.exp(w / self.hyps.temperature) for w in weights]
        exp_sum = sum(exps)
        normed_weights = [e / exp_sum for e in exps]
        
        assert abs(sum(normed_weights) - 1.0) < 1e-6, "Normalized weights do not sum to 1!"
        
        print("WEIGHTS FOR AGGREGATION: ", normed_weights)
        
        # Log aggregation weights to file
        weights_file = f"{self.hyps.results_dir}/aggregation_weights.txt"
        with open(weights_file, 'a') as f:
            weights_str = ','.join([f"{w:.6f}" for w in normed_weights])
            f.write(f"{weights_str}\n")
        
        parameters_per_client = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # weighted_weights = [
        #         [layer * normed_weights[i] for layer in parameters_per_client[i]] for i in range(len(parameters_per_client))
        #     ]
        
        weights_prime: NDArrays = [
            sum(parameters_per_client[client_idx][layer_idx] * normed_weights[client_idx] for client_idx in range(len(parameters_per_client)))
            for layer_idx in range(len(parameters_per_client[0]))
        ]

        # num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)
        # weights_prime: NDArrays = [
        #     reduce(np.add, layer_updates) / num_examples_total
        #     for layer_updates in zip(*weighted_weights)
        # ]
        
        parameters_aggregated = ndarrays_to_parameters(weights_prime)
        
         # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            pass
            # log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        
        new_global_prototype = Prototype.from_prototypes(prototypes, weights=normed_weights, plot=False)
        # Adapt the global prototype towards the new global prototype
        if self.prototype_adaptation > 0:
            self.global_prototype.adapt_towards(new_global_prototype, self.prototype_adaptation)
            # self.global_prototype.save(proto_path)
        
        
        return parameters_aggregated, metrics_aggregated
    
    def train_on_sampled_data(self):
        gpu = self.hyps.gpus[0]
        DEVICE = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        model = get_model(self.hyps, device=DEVICE)
        samples = self.global_prototype.sample(num_samples=self.num_samples, variance_scale=self.variance_scale)
        # Train the model on the sampled data
        
        
        
