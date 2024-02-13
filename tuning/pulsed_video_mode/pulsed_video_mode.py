"""
Created on 24/08/2023
@author jdh
"""
from template import *

import sys
# sys.path.remove( '/home/jdh/PycharmProjects/qgor')
sys.path.append('/home/jdh/PycharmProjects/qgor_qm')

from qgor.measurements.ramp2d_AWG.Waveforms.traditional_pulsing import make_ramps_and_sample_mask

pulse_amplitude = 0.1
fast_ramp_amplitude = 1
fast_frequency = 100
slow_ramp_amplitude = 1
pulse_samples = 50
fast_resolution = 10
slow_resolution = 5
n_pulses = fast_resolution * slow_resolution
measurement_samples = 100
slow_frequency = fast_frequency / slow_resolution
ratio = int(fast_frequency / slow_frequency)

slow, fast, mask = make_ramps_and_sample_mask(
        slow_ramp_amplitude,
        fast_ramp_amplitude,
        ratio,
        pulse_amplitude,
        -pulse_amplitude,
        slow_resolution,
        fast_resolution,
        pulse_samples,
        measurement_samples,
        ramp_type='stepped',
        acquisition_rearm_delay_samples=100
)


# pre filter compensation
plt.figure(figsize=(8, 4))
plt.plot(slow, label='Slow axis')
plt.plot(fast, label='Fast axis')
# plt.plot(mask, label='Mask')
plt.xlabel('AWG samples')
plt.ylabel('AWG output / V')
plt.legend()
plt.savefig('./pulsed_video_mode_with_mask.pdf')
plt.show()


