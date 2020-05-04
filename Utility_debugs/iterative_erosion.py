'''
input:
    --mask: input tumor mask
    --pct: the percentile of mask core
output:
    --mask_core: the mask core
'''
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt

def iterative_erosion(mask,pct):

    selem = np.ones((2 + 1, 2 + 1), dtype=np.uint8)
    mask_core=mask
    while np.sum(mask_core)/np.sum(mask)>pct:
        mask_core = morphology.binary_erosion(mask_core, selem)
    return mask_core