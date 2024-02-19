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

times = np.linspace(0, 1, 1000)



pulse = np.concatenate([
    np.zeros(0),
    np.ones(100),
    - np.ones(100),
    np.zeros(800)
])

measurex = times[200:300]
measure = np.zeros(100)

measure2 = times[500:600]

# possibly 2146 2020 1858

def process_2d(dictionary, data_name, x='dac10', y='dac12'):

    lims = [dictionary.get(x).get('data').min(), dictionary.get(x).get('data').max(), dictionary.get(y).get('data').min(), dictionary.get(y).get('data').max()]
    data = dictionary.get(data_name).get('data').T

    return data, lims


stab, stab_lims = process_2d(get_id(1944), 'square_difference_pulsed_unpulsed')
close, close_lims = process_2d(get_id(1947), 'square_difference_pulsed_unpulsed')

close *= 1e3
stab *= 1e3

normal, norm_lims = process_2d(get_id(1858), 'C', y='dac12')

fig, axs = plt.subplots(2, 2)
map = 'copper'
b = axs[0, 1].imshow(stab, origin='lower', aspect='auto', extent=stab_lims)
axs[1, 1].set_xlabel('DAC 10 / mV')
axs[0, 0].set_ylabel('DAC 12 / mV')


d = axs[1, 1].imshow(close, origin='lower', aspect='auto', extent=close_lims)
axs[1, 1].set_ylabel('DAC 12 / mV')


a = axs[0, 0].imshow(normal, origin='lower', aspect='auto', extent=norm_lims, cmap='hot')

c = axs[1, 0].plot(times, pulse, label='Pulse')
axs[1, 0].plot(measurex, measure, color='red', label='Measure')
axs[1, 0].plot(measure2, measure, color='red')
axs[1, 0].legend()
axs[1, 0].set_ylabel('Detuning ($\epsilon$) / (a. u.)')
axs[1, 0].set_xlabel('Proportion of pulse length')

plt.colorbar(a, ax=axs[0, 0], orientation='horizontal', location='top')
plt.colorbar(b, ax=axs[0, 1], orientation='horizontal', location='top')
plt.colorbar(d, ax=axs[1, 1], orientation='horizontal', location='top')



for ax in [axs[0,0], axs[0, 1], axs[1, 1]]:
    ax.set_aspect('equal', 'box')


# plt.tight_layout()
# plt.savefig('./difference_figure_i.svg')
# plt.savefig('./difference_figure_i.pdf')
plt.show()


