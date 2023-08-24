"""
Created on 28/06/2023
@author jdh
"""

import numpy as np

def load_np(npz_file):
    with open(npz_file, 'rb') as f:
        data = np.load(f)

        return dict(data.items())