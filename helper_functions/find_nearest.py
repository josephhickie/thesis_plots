"""
Created on 23/06/2023
@author jdh
"""

import numpy as np

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx