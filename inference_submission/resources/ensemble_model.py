import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Any
from .model import MultimodalFusionSimultaneousModel, MultimodalHierarchical

class EnsembleOfModels:
    def __init__(self, models, 
                 device: str,
                 ensemble_strategy: str = "average",                 
                 weights: Optional[List[float]] = None
                 ):
        self.models = models
        self.device = device
        self.ensemble_strategy = ensemble_strategy
        self.num_models = len(models)
        
        # Validate weights for weighted averaging
        if ensemble_strategy == "weighted_average":
            if weights is None:
                raise ValueError("Weights must be provided for weighted averaging")
            if len(weights) != self.num_models:
                raise ValueError("Number of weights must match number of models")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            self.weights = torch.tensor(weights, device=device)
        else:
            self.weights = None
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()

    @torch.no_grad()
    def predict(self, h: torch.Tensor, clinical_features: torch.Tensor, return_predictions:bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make ensemble predictions.
        
        Args:
            h: Image features tensor
            clinical_features: Clinical data tensor
            return_individual_predictions: Whether to return individual model predictions
            
        Returns:
            Ensemble prediction probabilities, optionally with individual predictions
        """
        # Ensure inputs are on correct device
        h = h.to(self.device)
        clinical_features = clinical_features.to(self.device)
        
        # Collect predictions from all models
        all_predictions = []
        all_logits = []
        
        for i, model in enumerate(self.models):
            try:
                model_results = model.forward(h, clinical_features)
                logits = model_results["MM"][0]
                prediction = model_results["MM"][1][-1]
                
                all_predictions.append(prediction)
                all_logits.append(logits)
                
            except Exception as e:
                print(f"Error during inference with model {i+1}: {e}")
                raise
        
        # Stack predictions
        stacked_predictions = torch.stack(all_predictions)  # Shape: (num_models, batch_size, num_classes)
        stacked_logits = torch.stack(all_logits)  # Shape: (num_models, batch_size, num_classes)
        
        # Apply ensemble strategy
        ensemble_prediction = self._combine_predictions(stacked_predictions, stacked_logits)
        
        if return_predictions:
            return ensemble_prediction, stacked_predictions
        
        return ensemble_prediction
    
    def _combine_predictions(self, predictions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Combine predictions using the specified ensemble strategy.
        
        Args:
            predictions: Stacked probability predictions (num_models, batch_size, num_classes)
            logits: Stacked logits (num_models, batch_size, num_classes)
            
        Returns:
            Combined prediction probabilities
        """
        
        if self.ensemble_strategy == "average":
            return torch.mean(predictions, dim=0)
        
        elif self.ensemble_strategy == "weighted_average":
            weights = self.weights.view(-1, 1, 1)  # Reshape for broadcasting
            return torch.sum(predictions * weights, dim=0)
        
        elif self.ensemble_strategy == "logit_average":
            avg_logits = torch.mean(logits, dim=0)
            return F.softmax(avg_logits, dim=1)
        
        elif self.ensemble_strategy == "majority_vote":            
            predicted_classes = torch.argmax(predictions, dim=1)  # (num_models,)
            
            # Count votes for each class (single sample)
            num_classes = predictions.shape[1]
            vote_counts = torch.zeros(num_classes, device=self.device)
            
            for j in range(self.num_models):
                predicted_class = predicted_classes[j]
                vote_counts[predicted_class] += 1
            
            # Convert vote counts to probabilities and add batch dimension
            probs = (vote_counts / self.num_models).unsqueeze(0)  # Shape: (1, num_classes)
            return probs[-1]

        elif self.ensemble_strategy in ["avg_maj_mix", "avg_maj_mix_80", "avg_maj_mix_90"]:
            predicted_classes = torch.argmax(predictions, dim=1)  # (num_models,)
            
            # Count votes for each class (single sample)
            num_classes = predictions.shape[1]
            vote_counts = torch.zeros(num_classes, device=self.device)
            
            for j in range(self.num_models):
                predicted_class = predicted_classes[j]
                vote_counts[predicted_class] += 1
            
            # Convert vote counts to probabilities and add batch dimension
            probs = (vote_counts / self.num_models).unsqueeze(0)  # Shape: (1, num_classes)
            prob = probs[-1]
            threshold = 0. if self.ensemble_strategy == "avg_maj_mix" else 0.8 if self.ensemble_strategy == "avg_maj_mix_80" else 0.9
            if prob[-1] > threshold and prob[-1] < 1 - threshold:
                # Use average if confidence is low
                return torch.mean(predictions, dim=0)
            else:
                # Use majority vote if confidence is high
                return prob
        
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.ensemble_strategy}")


class CrossValidationEnsemble(EnsembleOfModels):
    """
    Specialized ensemble for cross-validation models with identical architecture.
    """
    
    def __init__(self, 
                 base_model_config: Dict[str, Any],
                 cv_weights_paths: List[str],
                 device: str,
                 ensemble_strategy: str = "average",
                 ):
        """
        Initialize cross-validation ensemble.
        
        Args:
            base_model_config: Configuration for the base model architecture
            cv_weights_dir: Directory containing cross-validation model weights
            cv_fold_pattern: Pattern for CV weight filenames (e.g., "fold_{}.pt")
            num_folds: Number of CV folds
            ensemble_strategy: Strategy for combining predictions
            device: Device to run inference on
        """
        # Create config list (same config for all folds)
        num_folds = len(cv_weights_paths)
        model_configs = [base_model_config.copy() for _ in range(num_folds)]
        
        # Initialize with equal weights for all folds
        weights = [1.0 / num_folds] * num_folds if ensemble_strategy == "weighted_average" else None
        
        models = []
        for i, weights_path in enumerate(cv_weights_paths):
            
            # Initialize model with base config
            model = MultimodalFusionSimultaneousModel(**model_configs[i])
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    
            # Filter out instance_loss_fn related keys since we're using None for inference
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('clam.instance_loss_fn')}
            
            model.load_state_dict(filtered_state_dict, strict=False)
            
            model.to(device)
            models.append(model)
        
        super().__init__(
            models=models,
            ensemble_strategy=ensemble_strategy,
            weights=weights,
            device=device
        )
        
        self.num_folds = num_folds
        print(f"Initialized {num_folds}-fold cross-validation ensemble")



class CrossValidationEnsembleHierarchical(EnsembleOfModels):
    """
    Specialized ensemble for cross-validation models with identical architecture.
    """
    
    def __init__(self, 
                 base_model_config: Dict[str, Any],
                 cv_weights_paths: List[str],
                 device: str,
                 ensemble_strategy: str = "average",
                 ):
        """
        Initialize cross-validation ensemble.
        
        Args:
            base_model_config: Configuration for the base model architecture
            cv_weights_dir: Directory containing cross-validation model weights
            cv_fold_pattern: Pattern for CV weight filenames (e.g., "fold_{}.pt")
            num_folds: Number of CV folds
            ensemble_strategy: Strategy for combining predictions
            device: Device to run inference on
        """
        # Create config list (same config for all folds)
        num_folds = len(cv_weights_paths)
        model_configs = [base_model_config.copy() for _ in range(num_folds)]
        
        # Initialize with equal weights for all folds
        weights = [1.0 / num_folds] * num_folds if ensemble_strategy == "weighted_average" else None
        
        models = []
        for i, weights_path in enumerate(cv_weights_paths):
            
            # Initialize model with base config
            model = MultimodalHierarchical(**model_configs[i])
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            
            # Test if model has correct architecture for weights
    
            # Filter out instance_loss_fn related keys since we're using None for inference
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('clam.instance_loss_fn')}
            
            model.load_state_dict(filtered_state_dict, strict=False)
            
            
            
            
            model.to(device)
            models.append(model)
        
        super().__init__(
            models=models,
            ensemble_strategy=ensemble_strategy,
            weights=weights,
            device=device
        )
        
        self.num_folds = num_folds
        print(f"Initialized {num_folds}-fold cross-validation ensemble")
        
    @torch.no_grad()
    def predict(self, h_list: List[torch.Tensor], clinical_features: torch.Tensor, coords:List[torch.Tensor], return_predictions:bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make ensemble predictions.
        
        Args:
            h: Image features tensor
            clinical_features: Clinical data tensor
            return_individual_predictions: Whether to return individual model predictions
            
        Returns:
            Ensemble prediction probabilities, optionally with individual predictions
        """
        # Ensure inputs are on correct device
        # h = h.to(self.device)
        h_list = [d.to(self.device) for d in h_list]
        clinical_features = clinical_features.to(self.device)
        coords = [c.to(self.device) for c in coords]
        
        # Collect predictions from all models
        all_predictions = []
        all_logits = []
        
        for i, model in enumerate(self.models):
            try:
                # Get prediction from individual model
                # Add 1 dimension for batch processing
                clinical_features = clinical_features.unsqueeze(0) if clinical_features.dim() == 1 else clinical_features #!
                model_results = model.forward(h_list, clinical_features, coords)
                logits = model_results["MM"][0]
                prediction = model_results["MM"][1][-1]

                all_predictions.append(prediction)
                all_logits.append(logits)
                
            except Exception as e:
                print(f"Error during inference with model {i+1}: {e}")
                raise
        
        # Stack predictions
        stacked_predictions = torch.stack(all_predictions)  # Shape: (num_models, batch_size, num_classes)
        stacked_logits = torch.stack(all_logits)  # Shape: (num_models, batch_size, num_classes)
        
        # Apply ensemble strategy
        ensemble_prediction = self._combine_predictions(stacked_predictions, stacked_logits)
        
        if return_predictions:
            return ensemble_prediction, stacked_predictions
        
        return ensemble_prediction


class EnsembleOfEnsemble:
    def __init__(self, 
        model_classes: List[str],
        base_model_config: List[Dict[str, Any]],
        cv_weights_paths: List[List[str]],
        device: str,
        meta_ensemble_strategy: str = "average",
        ensemble_strategy: str = "average",
        ):
        self.model_classes = model_classes
        self.base_model_config = base_model_config
        self.cv_weights_paths = cv_weights_paths
        self.device = device
        self.meta_ensemble_strategy = meta_ensemble_strategy
        self.ensemble_strategy = ensemble_strategy
        
        self.models = []
        for i, model_class in enumerate(self.model_classes):
            if model_class == "hierarchical":
                model = CrossValidationEnsembleHierarchical(
                    base_model_config=self.base_model_config[i],
                    cv_weights_paths=self.cv_weights_paths[i],
                    device=self.device,
                    ensemble_strategy=self.ensemble_strategy
                )
            else:
                model = CrossValidationEnsemble(
                    base_model_config=self.base_model_config[i],
                    cv_weights_paths=self.cv_weights_paths[i],
                    device=self.device,
                    ensemble_strategy=self.ensemble_strategy
                )
            self.models.append(model)

    def _combine_predictions(self, ensemble_predictions: torch.Tensor) -> torch.Tensor:
        if self.meta_ensemble_strategy == "average":
            return torch.mean(ensemble_predictions, dim=0)
        elif self.meta_ensemble_strategy == "weighted":
            weights = self._get_model_weights()
            return torch.sum(weights[:, None, None] * ensemble_predictions, dim=0)
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.meta_ensemble_strategy}")

    @torch.no_grad()
    def predict(self, h_list: List[torch.Tensor], clinical_features: torch.Tensor, coords:List[torch.Tensor]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make ensemble predictions.

        Args:
            h: Image features tensor
            clinical_features: Clinical data tensor,
            coords: List of coordinate tensors,
            return_predictions: Whether to return individual model predictions

        Returns:
            Ensemble prediction probabilities, optionally with individual predictions
        """
        # Collect predictions from all models
        ensemble_predictions = []
        for i, model in enumerate(self.models):
            print(f"RUN MODEL {i+1}")
            if self.model_classes[i] == "hierarchical":
                ensemble_prediction, _ = model.predict(h_list, clinical_features, coords, return_predictions=True)
            else:
                ensemble_prediction, _ = model.predict(h_list[0], clinical_features, return_predictions=True)
            ensemble_predictions.append(ensemble_prediction)

        ensemble_predictions = torch.stack(ensemble_predictions)


        # Apply ensemble strategy
        ensemble_prediction = self._combine_predictions(ensemble_predictions)#, all_stacked_predictions)

        return ensemble_prediction