import timm
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
}

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out


def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []
	
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms

import os
def get_encoder(model_name, target_img_size=224, model_weights_path=None):
    uni_model_path = model_weights_path
    
    print(f"Loading model from: {uni_model_path}")
    if not os.path.exists(uni_model_path):
        raise FileNotFoundError(f"Model file not found: {uni_model_path}")
    
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    
    print("Creating model...")
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    
    print("Loading model weights...")
    model.load_state_dict(torch.load(uni_model_path, map_location="cpu"), strict=True)
    print("Model loaded successfully!")
    
    img_transforms = get_eval_transforms(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225],
                                         target_img_size = 224)
    
    return model, img_transforms

# def get_encoder(model_name, target_img_size=224, model_weights_path=None):
#     if model_name == 'resnet50_trunc':
#         model = TimmCNNEncoder()
#     elif model_name == 'uni_v1':
#         timm_kwargs = {
#             'model_name': 'vit_giant_patch14_224',
#             'img_size': 224, 
#             'patch_size': 14, 
#             'depth': 24,
#             'num_heads': 24,
#             'init_values': 1e-5, 
#             'embed_dim': 1536,
#             'mlp_ratio': 2.66667*2,
#             'num_classes': 0, 
#             'no_embed_class': True,
#             'mlp_layer': timm.layers.SwiGLUPacked, 
#             'act_layer': torch.nn.SiLU, 
#             'reg_tokens': 8, 
#             'dynamic_img_size': True
#         }
#         print("Creating model...")
#         model = timm.create_model(
#             pretrained=False, **timm_kwargs
#         )
#         print("Loading model weights...")
#         model.load_state_dict(torch.load(model_weights_path, map_location="cpu", weights_only=True), strict=True)

#     print("Model loaded successfully!1")
#     constants = MODEL2CONSTANTS[model_name]
#     img_transforms = get_eval_transforms(mean=constants['mean'],
#                                          std=constants['std'],
#                                          target_img_size = target_img_size)

#     return model, img_transforms