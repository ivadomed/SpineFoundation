import math

def make_lr_lambda(total_steps: int, warmup_steps: int, lr_up: float, lr_min: float):
    scale_min = lr_min / lr_up  # facteur minimal

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # warmup linéaire de lr_min -> lr_up
            return scale_min + (1.0 - scale_min) * (step / max(1, warmup_steps))

        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # cosine de lr_up -> lr_min
        return scale_min + 0.5 * (1.0 - scale_min) * (1.0 + math.cos(math.pi * t))

    return lr_lambda
