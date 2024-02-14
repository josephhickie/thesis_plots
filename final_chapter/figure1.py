import sys

import numpy as np
from pathlib import Path
# from template import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dir = '/Users/joseph/Documents/data_for_thesis'



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


if __name__ == '__main__':
    counts = get_id(1811)
    x = 'dac10'
    y = 'dac13'
    lims = [counts.get(x).get('data').min(), counts.get(x).get('data').max(), counts.get(y).get('data').min(), counts.get(y).get('data').max()]
    counts_data = counts.get('num_states').get('data').T



    # plt.figure()
    # plt.imshow(counts.get('num_states').get('data').T, origin='lower', extent=lims)
    # plt.xlabel('DAC 10')
    # plt.ylabel('DAC 13')
    # plt.show()


    peaks = get_id(1797)
    #
    # plt.figure()
    # plt.plot(peaks.get('x').get('data'), peaks.get('plot').get('data'))
    # plt.show()



    stability = get_id(2472)

    stab_data, stab_lims = process_2d(stability, 'C', x='dac9', y='comp12')
    stab_lims[3] = stab_lims[3] + 40

    # plt.figure()
    # plt.imshow(stab_data, origin='lower', extent=stab_lims)
    # plt.xlabel('DAC 10')
    # plt.ylabel('DAC 13')
    # plt.show()


    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1]})

    p = axs[0, 0].imshow(stab_data, origin='lower', extent=stab_lims, cmap='hot')
    axs[0, 0].set_xlabel('DAC 10 / mV')
    axs[0, 0].set_ylabel('DAC 13 / mV')
    axs[0, 1].set_xlabel('DAC 10 / mV')
    axs[0, 1].set_ylabel('DAC 13 / mV')

    axs[1, 0].hist(stab_data.flatten(), bins=80, color='black')
    axs[1, 0].set_ylabel('Counts')
    axs[1, 0].set_xlabel('$V_\mathrm{rf}$')

    axs[1, 1].plot(peaks.get('x').get('data'), peaks.get('plot').get('data') / np.max(peaks.get('plot').get('data')), color='black')
    axs[1, 1].set_xlabel('Pixel')
    axs[1, 1].set_ylabel('$\\nabla V_\mathrm{rf} \, (a. u.)$')


    plot = axs[0, 1].imshow(counts_data, origin='lower', extent=lims)
    plt.colorbar(plot, ax=axs[0, 1], orientation='horizontal', location='top')
    plt.colorbar(p, ax=axs[0, 0], orientation='horizontal', location='top')
    fig.tight_layout()
    plt.savefig('./counts.svg')
    plt.show()