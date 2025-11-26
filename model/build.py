"""
This script centralises model building.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import importlib


AVAILABLE_MODELS = {
    "cnn3d": ("model.CNNencoder", "CNN3DFeatureExtractor"),
    "spine_encoder": ("model.SpineEncoder", "SpineEncoder"),
    "spine_mae": ("model.SpineMAE", "SpineMAE"),
}



def build_model(name, params=None):
    key = name.lower()
    if key not in AVAILABLE_MODELS:
        raise KeyError(f"Unknown model '{name}'")

    module_path, class_name = AVAILABLE_MODELS[key]

    params = params or {}
    if not isinstance(params, dict):
        raise TypeError('params must be a dict of constructor keyword arguments')

    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)
    print("\n========== MODEL ==========")
    print(f"Building model: {name}")
    for k, v in params.items():
        print(f"  - {k:15} = {v}")
    return ModelClass(**params)


if __name__ == '__main__':

	print('Available models:', ', '.join(AVAILABLE_MODELS))

	m = build_model('spine_encoder', {'img_size': (32, 32, 32), 'patch_size': (8, 8, 8), 'embed_dim': 64, 'num_layers': 2,"num_heads":4, "mlp_dim":128})
	print('Built model:', type(m), 'params:', sum(p.numel() for p in m.parameters()))
