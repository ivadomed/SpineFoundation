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
    "swin_mae": ("model.swin_mae", "SwinMAE3D"),
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

    import torch

    def can_train_step(model, shape, device="cuda"):
        try:
            model.train().to(device)
            x = torch.randn(shape, device=device)
            y = model(x)
            loss = y.square().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            del x, y, loss
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False


    def find_max_cube_train(model, in_channels=1, lo=32, hi=512, step=32, device="cuda"):
        lo = (lo + step - 1) // step * step
        hi = hi // step * step
        ok = None
        while lo <= hi:
            mid = ((lo + hi) // (2 * step)) * step
            shape = (1, in_channels, mid, mid, mid)
            if can_train_step(model, shape, device=device):
                ok = mid
                lo = mid + step
            else:
                hi = mid - step
        return ok

# usage:
# max_size = find_max_cube_train(model, in_channels=1, lo=32, hi=512, step=32)
# print("max cube train size:", max_size)

    file ="./model/SwinUNETR.json"
    import json

    with open(file,"r") as f:
        config = json.load(f)
    config.pop("model_name", None)
    config.pop("img_resolution", None)
    model = build_model("swin_unetr", params=config).to("cuda")

    max_size = find_max_cube_train(model, in_channels=1, lo=32, hi=512, step=32)
    print("max cube train size:", max_size)
    """for block in model.decoder.blocks:
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
            break"""



