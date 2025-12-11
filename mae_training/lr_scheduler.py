import math

def make_lr_lambda(total_steps: int, warmup_steps: int, lr_up: float, lr_min: float):
    """
    Scheduler:
      1. Warmup linéaire       [0 → warmup_steps]       : lr_min → lr_up
      2. Plateau constant      [warmup → 50% steps]     : lr_up
      3. Cosine decay          [50% steps → total]      : lr_up → lr_min
    """

    plateau_end = int(0.50 * total_steps)  # 50% du training après warmup

    def lr_lambda(step: int) -> float:

        # --- 1. WARMUP ---
        if step < warmup_steps:
            # interpolation linéaire entre lr_min et lr_up
            alpha = step / warmup_steps
            return (lr_min + alpha * (lr_up - lr_min)) / lr_up

        # --- 2. PLATEAU ---
        if step < plateau_end:
            return 1.0  # lr = lr_up

        # --- 3. COSINE DECAY ---
        # interpolation cosinus entre lr_up et lr_min
        decay_steps = total_steps - plateau_end
        t = (step - plateau_end) / decay_steps  # ∈ [0,1]
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        lr_now = lr_min + (lr_up - lr_min) * cosine
        return lr_now / lr_up

    return lr_lambda
