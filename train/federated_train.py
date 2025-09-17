import os
import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar

from flwr.server import ServerApp
from flwr.client import ClientApp
from flwr.simulation import run_simulation

from server_app import server_config
from client_app import client_config
from hyperparameters import get_hyperparameters
disable_progress_bar()

def seed_torch(device, seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_everything(seed):
    """Comprehensive seeding for all randomness sources"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables for deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_experiment(device, args):
    seed = args.seed
    seed_torch(device, seed)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)


    args.split_dir = os.path.join('splits', args.split_dir)

    assert os.path.isdir(args.split_dir), "Split directory does not exist: {}".format(args.split_dir)


def run_experiment(args):
    # Set environment variables early for maximum consistency with MVP
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    # Seed everything before any operations
    seed_everything(args.seed)
    
    gpu = args.gpus[0]
    # Set CUDA environment if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    DEVICE = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    NUM_CLIENTS = args.num_clients
    BATCH_SIZE = 32

    # Create Ray temp directory in scratch space
    ray_temp_dir = "/local/scratch/phempel/ray_tmp"
    os.makedirs(ray_temp_dir, exist_ok=True)
    
    # Set RAY_TMPDIR environment variable to force Ray to use our temp directory
    os.environ["RAY_TMPDIR"] = ray_temp_dir
    
    # Configure Ray backend to allocate GPU resources to clients
    backend_config = {
        "client_resources": {"num_cpus": 1, "num_gpus": 0.3},  # Allocate 1 GPU per client since we only have 1 client
        "runtime_env": {
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", str(gpu)),
                "PYTHONHASHSEED": str(args.seed),
                "CUBLAS_WORKSPACE_CONFIG": ":16:8"
            }
        }
    }
    init_experiment(device=DEVICE, args=args)

    run_simulation(
        server_app=ServerApp(server_fn=server_config(args=args)),
        client_app=ClientApp(client_fn=client_config(args=args)),
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    
    orig_args = get_hyperparameters()
    
    if orig_args.multi_seed is not None:
        for seed in orig_args.multi_seed:
            args = orig_args  # Use a separate variable to avoid modifying the original args
            args.seed = seed
            print(f"Running experiment with seed {seed}")
            run_experiment(args)
    else:
        import time
        start_time = time.time()
        run_experiment(orig_args)
        end_time = time.time()
        print(f"Experiment completed in {end_time - start_time:.2f} seconds")
    
    
    
# python federated_train.py --gpus 1 --num_clients 3 --exp_code 3_clients --no_verbose --split_dir chimera_3_0.1 --num_rounds 3 --seed 1