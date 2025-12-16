"""
This script centralises model building.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import importlib


AVAILABLE_MODELS = {
    "spine_encoder": ("model.SpineEncoder", "SpineEncoder"),
    "spine_mae": ("model.SpineMAE", "SpineMAE"),
    "spine_vit_seg": ("model.decoderseg", "SpineSeg"),
    "spine_maev2": ("model.SpineMAEv2", "SpineMAE"),
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


if __name__ == "__main__":
    file ="./model/SpineMAE.json"
    import json

    with open(file,"r") as f:
        config = json.load(f)
    config.pop("model_name", None)
    config.pop("img_resolution", None)
    model = build_model("spine_mae", params=config).to("cuda")

    for block in model.decoder.blocks:
        for param in block.cross_attn.parameters():
            param.requires_grad = False
        for param in block.norm_cross_attn.parameters():
            param.requires_grad = False

    from torchinfo import summary
    summary(model, input_size=[1,1,96, 96, 320],device="cuda")
    print(model.decoder)               # ou model.SpineDecoder selon ton code
    print(model.decoder.blocks[0])     # inspecte le MLP
    # et/ou
    for n,p in model.named_parameters():
        if "decoder" in n and "mlp" in n and "weight" in n:
            print(n, p.shape)
            break



