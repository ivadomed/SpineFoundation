"""
This script centralises model building.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import importlib


AVAILABLE_MODELS = {
    "spine_encoder": ("model.SpineEncoder", "SpineEncoder"),
    "spine_mae": ("model.SpineMAE", "SpineMAE"),
    "spine_vit_seg": ("model.seg", "SpineViTSeg"),
}



def build_model(name, params=None,rank=0):
    key = name.lower()
    if key not in AVAILABLE_MODELS:
        raise KeyError(f"Unknown model '{name}'")

    module_path, class_name = AVAILABLE_MODELS[key]

    params = params or {}
    if not isinstance(params, dict):
        raise TypeError('params must be a dict of constructor keyword arguments')

    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)
    if rank==0:
        print("\nMODEL :\n")
        print(f"Building model: {name}")
    #for k, v in params.items():
     #   print(f"  - {k:15} = {v}")
    return ModelClass(**params)


