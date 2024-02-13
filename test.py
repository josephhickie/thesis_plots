import numpy as np


P = np.array([-1, -2, -3])
Q = np.array([-3, -3, -5])

def calculate_for_any(P, Q, x=None, y=None, z=None):

    displacement_vector = Q - P

    r = lambda t: P + t * displacement_vector

    # r(t) = Q + t * displacement_vector

    


    t = (z - P[2]) / displacement_vector[2]

    return r(t)





