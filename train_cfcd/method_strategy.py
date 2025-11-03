

from tkinter.messagebox import WARNING
from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from matplotlib.pylab import log
from utils.method_utils import Prototype
import numpy as np
from functools import  reduce

from flwr.common import (
    parameters_to_ndarrays,
    NDArrays,
    ndarrays_to_parameters
)

class CustomFedAvg(FedAvg):

    def __init__(self, prototype_adaptation: float = 0, variance_scale: float = 1.0, num_samples=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prototype_adaptation = prototype_adaptation
        self.global_prototype = None
        self.variance_scale = variance_scale
        self.num_samples = num_samples

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy | FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str,bool |bytes | float | int | str]]:
        if server_round == 1:
            
            prototypes = []
            for client_proxy, fit_res in results:
                proto = fit_res.metrics.get("prototype", None)
                if proto is not None:
                    prototype = Prototype.deserialize(proto)
                    prototypes.append(prototype)
                else:
                    raise ValueError("Client did not return a prototype!")
            
            global_prototype = Prototype.from_prototypes(prototypes, plot=True)
            self.global_prototype = global_prototype
            
        distances = []
        prototypes = []
        for client_proxy, fit_res in results:
            proto = fit_res.metrics.get("prototype", None)
            if proto is not None:
                prototype = Prototype.deserialize(proto)
                prototypes.append(prototype)
            else:
                raise ValueError("Client did not return a prototype!")
            distance = self.global_prototype.distance_prototype(prototype)
            distances.append(distance)
        
        # Normalize distances to distribution weights
        distance_sum = sum(distances)
        if distance_sum == 0:
            raise ValueError("All client prototypes are identical to the global prototype!")
        else:
            normed_distances = [d / distance_sum for d in distances]
        # Invert distances to get weights (closer clients have higher weight)
        weights = [1 - d for d in normed_distances]
        weight_sum = sum(weights)
        normed_weights = [w / weight_sum for w in weights]
        assert abs(sum(normed_weights) - 1.0) < 1e-6, "Normalized weights do not sum to 1!"
        
        parameters_per_client = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        weighted_weights = [
                [layer * normed_weights[i] for layer in parameters_per_client[i]] for i in range(len(parameters_per_client))
            ]
        
        num_examples_total = sum( fit_res.num_examples for (_, fit_res) in results)
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        
        parameters_aggregated = ndarrays_to_parameters(weights_prime)
        
         # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            pass
            # log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        
        new_global_prototype = Prototype.from_prototypes(prototypes, weights=normed_weights, plot=True)
        # Adapt the global prototype towards the new global prototype
        if self.prototype_adaptation > 0:
            self.global_prototype.adapt_towards(new_global_prototype, self.prototype_adaptation)
        
        
        return parameters_aggregated, metrics_aggregated
    
    def train_on_sampled_data(self):
        samples = self.global_prototype.sample(num_samples=self.num_samples, variance_scale=self.variance_scale)
        
