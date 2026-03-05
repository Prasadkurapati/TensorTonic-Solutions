import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x, dtype=float)
    if rng is not None:
        rand = rng.random(x.shape)
    else:
        rand = np.random.random(x.shape)
    scale = 1.0 / (1.0 - p)
    pattern = (rand < (1 - p)).astype(float) * scale
    output = x * pattern
    return (output, pattern)