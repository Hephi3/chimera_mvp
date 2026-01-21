"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_PATH = Path("/opt/ml/model")  # Path where Grand Challenge extracts model tarball
TMP_PATH = Path("/tmp/")  # Temporary path for intermediate files


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "bladder-cancer-tissue-biopsy-whole-slide-image",
            "chimera-clinical-data-of-bladder-cancer-patients",
            "tissue-mask",
        ): my_method,
    }[interface_key]

    # Call the handler
    return handler()

def my_method():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #### Prepare data ####
        try:
            cd_data = load_json_file(location=INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-patients.json")
            from resources.preprocessing_cd import preproc_cd_file
            cd_preprocessed = preproc_cd_file(cd_data)
            # Flatten the clinical data
            cd_features = []
            for key in cd_preprocessed:
                if isinstance(cd_preprocessed[key], list):
                    cd_features.extend(cd_preprocessed[key])
                else:
                    cd_features.append(cd_preprocessed[key])
        #Every other error:
        except Exception as e:
            print(f"Unexpected error while processing clinical data: {e}")
            cd_features = [0] * 12  # Assuming 12 features as per CD_KEYS
            print(f"Using fallback clinical data features: {cd_features}")
        
        
        # Find the actual image and mask files
        image_files = glob(str(INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi/*.tif"))
        mask_files = glob(str(INPUT_PATH / "images/tissue-mask/*.tif"))
        
        if not image_files:
            raise FileNotFoundError("No image files found in the expected directory.")
        if not mask_files:
            mask_files = None
    
    
        uni_model_path = MODEL_PATH / "pytorch_model.bin"
        if not uni_model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {uni_model_path}.")
        
        stain_template_path = RESOURCE_PATH / "stain_template.png"
        if not stain_template_path.exists():
            raise FileNotFoundError(f"Stain template not found at {stain_template_path}.")
        
        
        from resources.main_create_features import create_features_hiearchical
        img_features_list, coords = create_features_hiearchical(
            wsi_path=image_files[0],  # Use the first image file
            output_dir= TMP_PATH / "features",  # Save features to temporary directory
            mask_path=mask_files[0] if mask_files else None,  # Use the first mask file if available
            stain_template_path=stain_template_path,  # Use the stain template from resources
            model_weights_path=uni_model_path,  # Use the model weights from model path
            device=device,  # Use the detected device (CPU or GPU)
        )
        
        # To device
        img_features_list = [
            img_features.to(device=device, dtype=torch.float32) for img_features in img_features_list
        ]
        cd_features = torch.tensor(cd_features, device=device, dtype=torch.float32)
        coords = [c.to(device=device, dtype=torch.float32) for c in coords]   
        
        
        ### MODEL ###
        model_names = ["Model1", "Model2", "Model3"]
        model_classes = ["hierarchical", "hierarchical", "hierarchical"]
        model_configs = [
            {'dropout': 0.4, 'n_classes': 3, 'embed_dim': 1536, 'size_arg': 'tiny', 'subtyping': True, 'k_sample': 8, 'instance_loss_fn': None, 'num_levels': 3, 'clinical_dim': 256, 'norm': True, 'use_auxiliary_loss': False},
            {'dropout': 0.4, 'n_classes': 3, 'embed_dim': 1536, 'size_arg': 'tiny', 'subtyping': True, 'k_sample': 8, 'instance_loss_fn': None, 'num_levels': 3, 'clinical_dim': 256, 'norm': True, 'use_auxiliary_loss': False},
            {'dropout': 0.5, 'n_classes': 3, 'embed_dim': 1536, 'size_arg': 'tiny', 'subtyping': True, 'k_sample': 8, 'instance_loss_fn': None, 'num_levels': 3, 'clinical_dim': 256, 'norm': True, 'use_auxiliary_loss': False},

        ]

        from resources.ensemble_model import EnsembleOfEnsemble
        
        ensemble = EnsembleOfEnsemble(
            model_classes=model_classes,
            base_model_config=model_configs,
            cv_weights_paths=[[MODEL_PATH / model_names[i] / f"s_{j}_checkpoint.pt" for j in range(10)] for i in range(len(model_classes))],
            device=device,
            meta_ensemble_strategy="average",
            ensemble_strategy="majority_vote"
        )
        
        with torch.no_grad():
            prob = ensemble.predict(img_features_list, cd_features, coords)


        print(f"Prediction: {prob}")
        prob = prob.cpu().numpy().tolist()[-1]
        print(f"->Prediction: {prob}")
        # Save your output
        write_json_file(
            location=OUTPUT_PATH / "brs-probability.json",
            content=prob,
        )
    
        return 0
    except Exception as e:
        write_json_file(
            location=OUTPUT_PATH / "brs-probability.json",
            content=0
        )
        return 1


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))



if __name__ == "__main__":
    raise SystemExit(run())
