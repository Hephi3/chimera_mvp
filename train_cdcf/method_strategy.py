

from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from utils.method_utils import Prototype

class CustomFedAvg(FedAvg):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_prototype = None

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy | FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str,bool |bytes | float | int | str]]:
        if server_round == 1:
            
            prototypes = []
            for client_proxy, fit_res in results:
                proto = fit_res.metrics.get("prototype", None)
                if proto is not None:
                    prototype = Prototype.deserialize(proto)
                    prototypes.append(prototype)
            
            global_prototype = Prototype.from_prototypes(prototypes, plot=True)
            print(1/0)
            self.global_prototype = global_prototype

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        return parameters_aggregated, metrics_aggregated
