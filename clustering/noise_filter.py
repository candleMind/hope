import numpy as np

from clustering.similarity import seed_max


def keep_mask(X, seeds, thr):
    keep = seed_max(X, seeds) > thr
    return keep, float(np.mean(keep))
