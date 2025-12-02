import math

def make_lr_lambda(total_steps: int, warmup_steps: int, lr_up: float, lr_min: float):
    scale_min = lr_min / lr_up  # facteur minimal

    def lr_lambda(step: int) -> float:
        if step < 2200:
            # warmup linéaire de lr_min -> lr_up
            return 1
        elif 2200 <= step :
            return 8e-3
        return 4e-3

    return lr_lambda
