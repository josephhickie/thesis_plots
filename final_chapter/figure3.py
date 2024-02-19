"""
Created on 13/02/2024
@author jdh
"""


import sys
sys.path.append('/home/jdh/PycharmProjects/fine_tuning/')


import numpy as np
from pathlib import Path
# from template import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from fine_tuning.detection import get_triple_point_idx, count_lines, plot_peaks, identify, get_points, kde_detection
from fine_tuning.detection import transition_ids
dir = '/home/jdh/Documents/data_for_thesis'



def get_measurement(folder):
    folder = folder if isinstance(folder, Path) else Path(folder)

    out = {}

    for file in folder.glob("*.dat"):
        name, unit, memmap = open_memmap(file_path=file)
        out[name] = {'data': np.array(memmap), 'unit': unit}

    return out
def open_memmap(file_path: Path, mode="r"):
    """
    A function to open a memory map, while also parsing its name to retrieve the name, units, data_type and array shape.
    @param file_path: pathlib path to the folder.
    @param mode: the mode with which to open the memmap.
    @return: name, unit, memmap.
    """

    name, unit, data_type, shape_string = file_path.stem.split("-", 3)
    # clipping the brackets off
    shape_string = shape_string[1:-1]

    # parsing the shape string
    shape = tuple(shape_string.split(",", shape_string.count(",")))
    shape = tuple(filter(lambda x: x != "", shape))
    shape = tuple(int(i) for i in shape)

    # loading the memmaps
    memmap = np.memmap(filename=file_path, dtype=data_type, mode=mode, shape=shape)

    return name, unit, memmap

get_id = lambda id: get_measurement(dir + '/' + str(id))

def process_2d(dictionary, data_name, x='dac10', y='dac13'):

    lims = [dictionary.get(x).get('data').min(), dictionary.get(x).get('data').max(), dictionary.get(y).get('data').min(), dictionary.get(y).get('data').max()]
    data = dictionary.get(data_name).get('data').T

    return data, lims

def trim_data(data, trim_idx):

    return data[trim_idx:-trim_idx, trim_idx:-trim_idx]

data, _ = process_2d(get_id(2472), 'D', x='dac9', y='comp12')
data = trim_data(data, 3)
plt.imshow(data.T, origin='lower')





peaks_counted, idx, traces = count_lines(data)
points = get_points(data)
plot_peaks(data)

transitions = transition_ids(data, search_point_fraction=0.1)

plt.figure()
plt.imshow(data.T, origin='lower')#
plt.scatter(transitions[:, 0], transitions[:, 1], color='red')
plt.show()