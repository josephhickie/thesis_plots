"""
Created on 23/06/2023
@author jdh

these are helper_functions taken from readout_opt which i should just import but setting up the jax library again is a pain
"""

import sys

from helper_functions import find_nearest_idx
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots
import matplotlib


import math
from scipy.special import erf

from scipy.integrate import quad

plt.style.use(['science', 'ieee', 'bright', 'grid', 'no-latex'])
font = {'size': 10}
matplotlib.rc('font', **font)

def gaussian(x, location, scale):

    return (1 / (scale * np.sqrt(2. * np.pi))) * np.exp((-1/2) * ((x - location)**2) / scale**2)

def _normalise_weights(w1, w2, w3):
    "normalise weights so the full probability distribution still sums to 1"


    sum = w1 + w2 + w3

    return w1/sum, w2/sum, w3/sum



def triplet_decay_probability(v_rf, tau, v_rf_s, v_rf_t, sigma, P_t):
    # tau is integration time / T1 relaxation time
    # normalise the distribution function by dividing by the integral over basically the full range.
    # -10 to 10 is enough to get the full range.

    out = _unnorm_prob(v_rf, tau, v_rf_s, v_rf_t, sigma, P_t) / (
        (_integral(10, tau, v_rf_s, v_rf_t, sigma, P_t) - _integral(-10, tau, v_rf_s, v_rf_t, sigma, P_t))
    )

    # added machine precision to avoid nans
    return np.where(np.isnan(out), np.zeros_like(out), out) + (2.**-31)


def _unnorm_prob(value, tau, v_rf_s, v_rf_t, sigma, P_t):
    """
    triplet decay distribution (not normalise)
    :param value: returns probability density at P(x=value)
    :return: P(x=Value) (can broadcast)
    """

    delta_v = v_rf_t - v_rf_s
    sqrd_sigma = sigma * sigma
    sqrt2 = np.sqrt(2.)

    # from https://arxiv.org/pdf/2203.06608.pdf
    unnormalised_probability_function = (tau * P_t / (np.sqrt(2 * np.pi) * delta_v)) * np.exp(
        (tau / delta_v) * (v_rf_s - value + (tau * sqrd_sigma / (2. * delta_v)))
    ) * (
                                                erf(
                                                    (((sigma * tau) / (delta_v * sqrt2)) + (
                                                            (v_rf_t - value) / (sqrt2 * sigma)))
                                                )
                                                - erf(
                                            ((sigma * tau) / (delta_v * sqrt2)) + (
                                                    (v_rf_s - value) / (sigma * sqrt2))
                                        )
                                        )

    return unnormalised_probability_function

def _integral(x, tau, v_rf_s, v_rf_t, sigma, P_t):

    """
    Calculates the value of the integral of the analytical triplet model at x. Used to calculate cdf so we make the
    distribution a pdf (integrates to 1).

    :param x: value at which to evaluate the integral (v_rf)
    :param tau: integration time / T1
    :param v_rf_s: singlet readout point
    :param v_rf_t: triplet readout point
    :param sigma: noise
    :param P_t: triplet probability
    :return:
    """

    # https://www.wolframalpha.com/input?i=integrate+a+*+exp%28b+*+%28s+-+x+%2B+c%29%29+*+%28erf%28d+%2B+%28%28t+-+x%29+%2F+f%29%29+-+erf%28d+%2B+%28%28s+-+x%29+%2F+f%29%29%29+

    sqrd_sigma = sigma * sigma
    sqrt2 = np.sqrt(2.)

    delta_V = v_rf_t - v_rf_s
    a = tau * P_t / (np.sqrt(2. * np.pi) * delta_V)
    b = tau / delta_V
    c = tau * sqrd_sigma / (2. * delta_V)
    d = sigma * tau / (delta_V * sqrt2)
    f = sigma * sqrt2
    t = v_rf_t
    s = v_rf_s

    x1 = a / b
    x2 = np.exp(-b * (-c + (d * f) + t + x))
    x3 = np.exp(b / 4. * (b * f ** 2. + (4. * (t + x))))
    x4 = erf(- b * f / 2 + d + (s / f) - (x / f))
    x5 = np.exp(b / 4. * (b * f ** 2. + (4. * (s + x))))
    x6 = math.erf(- b * f / 2 + d + (t / f) - (x / f))
    x7 = np.exp(b * (d * f + s + t)) # equating to zero
    x8 = math.erf((d * f + s - x) / f)
    x10 = math.erf((d * f + t - x) / f)

    equation = x1 * x2 * ((-x3 * x4) + (x5 * x6) + (x7 * (x8 - x10)))

    return equation


def full_parametric_probability(v_rf, v_rf_s, v_rf_t, sigma, w1, w2, w3):


    w1, w2, w3 = _normalise_weights(np.abs(w1), np.abs(w2), np.abs(w3))
    P_t = w2 + w3
    singlet, triplet, triplet_decay = separated_parametric_probability(v_rf, v_rf_s, v_rf_t, sigma, w2, w3, P_t)

    return w1 * singlet + w2 * triplet + w3 * triplet_decay


def separated_parametric_probability(v_rf, v_rf_s, v_rf_t, sigma, w2, w3, P_t, tau=None):

    # the weights are ratios so it doesn't matter if they have been normalised here or not
    if tau is None:
        tau = - np.log(1 / (1 + ((w3) / (w2))))

    singlet = gaussian(v_rf, v_rf_s, sigma)
    triplet = gaussian(v_rf, v_rf_t, sigma)
    triplet_decay = triplet_decay_probability(v_rf, tau, v_rf_s, v_rf_t, sigma, P_t)

    return singlet, triplet, triplet_decay

# as above but just returns as helper_functions (for integration - debugging)
def separated_parametric_probability_functions(v_rf_s, v_rf_t, sigma, w2, w3, P_t, tau=None):

    # the weights are ratios so it doesn't matter if they have been normalised here or not
    if tau is None:
        tau = - np.log(1 / (1 + ((w3) / (w2))))

    singlet = lambda v_rf: gaussian(v_rf, v_rf_s, sigma)
    triplet = lambda v_rf: gaussian(v_rf, v_rf_t, sigma)
    triplet_decay = lambda v_rf: triplet_decay_probability(v_rf, tau, v_rf_s, v_rf_t, sigma, P_t)

    return singlet, triplet, triplet_decay



def each_parametric_part(v_rf, v_rf_s, v_rf_t, sigma, w1, w2, w3, tau=None):


    # w1, w2, w3 = _normalise_weights(np.abs(w1), np.abs(w2), np.abs(w3))

    P_t = (w2 + w3)

    singlet, triplet, triplet_decay = separated_parametric_probability(v_rf, v_rf_s, v_rf_t, sigma, w2, w3, P_t, tau)
    return w1 * singlet, w2 * triplet, w3 * triplet_decay


def full_log_probability(v_rf, v_rf_s, v_rf_t, sigma, w1, w2, w3):
    w1, w2, w3 = _normalise_weights(np.abs(w1), np.abs(w2), np.abs(w3))

    P_t = (w2 + w3)

    return np.log(full_parametric_probability(v_rf, v_rf_s, v_rf_t, sigma, w1, w2, w3))



def deltas_at(v0, v1, vs):

    idx0 = find_nearest_idx(vs, v0)
    idx1 = find_nearest_idx(vs, v1)

    out = np.zeros_like(vs)
    out[idx0] = 10
    out[idx1] = 10

    return out

def exponential_decay_dist(v0, v1, vs, rate):

    pdf = rate * np.exp(-rate * vs)

    # mask
    pdf[vs < v0] = 0
    pdf[vs > v1] = 0

    area = exp_cdf(v1, rate) - exp_cdf(v0, rate)

    return pdf / area

def exp_cdf(v, rate):
    return 1 - np.exp(-rate * v)


v_rfs = np.linspace(0, 2, 2000)
v0 = 0.7
v1 = 1.3

FIGSIZE = (5, 2.5)

deltas = deltas_at(v0, v1, v_rfs)

plt.figure(figsize=FIGSIZE)
plt.plot(v_rfs, deltas)
plt.show()

rate = .5
noise_scale = 0.1
gaussian_centre = 1

exp = exponential_decay_dist(v0, v1, v_rfs, rate)


plt.figure(figsize=FIGSIZE)
plt.plot(v_rfs, exp)
plt.show()


noise = gaussian(v_rfs, gaussian_centre, noise_scale)

plt.figure(figsize=FIGSIZE)
plt.plot(v_rfs, noise)
plt.show()


noise_meas = np.convolve(deltas, noise, mode='same')
noise_decay = np.convolve(exp, noise, mode='same')

plt.figure()
plt.plot(v_rfs, noise_meas)
plt.show()

plt.figure()
plt.plot(v_rfs, noise_meas)
plt.plot(v_rfs, deltas)
plt.show()

plt.figure(figsize=FIGSIZE)
plt.plot(v_rfs, noise_decay)
plt.show()

plt.figure(figsize=FIGSIZE)
plt.plot(v_rfs, noise_decay/np.trapz(noise_decay, v_rfs))
plt.show()

