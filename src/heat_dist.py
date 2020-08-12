"""
heat_transfer/src/heat_dist.py
"""

import numpy as np


def transfer2D(density=7600,
               conduct=47,
               capacity=480,
               t0=400,
               tt=800,
               tb=300,
               tl=200,
               tr=500,
               m=1,
               L=5,
               n=50,
               ft=500
               ):
    """
    Generate 2D heat transfer2D data.

    Args:
        density: density (kg/m3)
        conduct: heat conductivity (W/m/K)
        capacity: heat capacity (K)
        t0: initial temperature (K)
        tt: Top edge temperature (K)
        tb: Bottom edge temperature (K)
        tl:  Left edge temperature (K)
        tr: Right edge temperature (K)
        m: Module
        L : Edge length (m)
        n: Number of points
        ft: Final time (s)

    Returns:
        Matrix of NxN of the heat transfer.

    """

    # calculation
    a = conduct / density / capacity
    # length step
    dx = m / (n - 1)
    # time step
    dt = (dx ** 2) / L / a
    t = dt

    # initial temperatures
    heat = np.zeros((n, n))
    for i in range(n):
        for j in range(1, n):
            heat[i, j] = t0

    while t < ft:

        j, i = 0, 0
        heat[i, j] = (heat[i + 1, j] + tr + tb + heat[i, j + 1] + (L - 4) * heat[i, j]) / L

        j, i = n - 1, 0
        heat[i, j] = (heat[i + 1, j] + tb + heat[i, j - 1] + tl + (L - 4) * heat[i, j]) / L

        j, i = 0, n - 1
        heat[i, j] = (tt + heat[i - 1, j] + tr + heat[i, j + 1] + (L - 4) * heat[i, j]) / L

        j, i = n - 1, n - 1
        heat[i, j] = (tl + heat[i - 1, j] + heat[i, j - 1] + tl + (L - 4) * heat[i, j]) / L

        i = 0
        for j in range(0, n - 1):
            heat[i, j] = (heat[i + 1, j] + tb + heat[i, j - 1] + heat[i, j + 1] + (L - 4) * heat[i, j]) / L

        i = n - 1
        for j in range(0, n - 1):
            heat[i, j] = (tt + heat[i - 1, j] + heat[i, j - 1] + heat[i, j + 1] + (L - 4) * heat[i, j]) / L

        j = 0
        for i in range(0, n - 1):
            heat[i, j] = (heat[i + 1, j] + heat[i - 1, j] + tr + heat[i, j + 1] + (L - 4) * heat[i, j]) / L

        j = n - 1
        for i in range(0, n - 1):
            heat[i, j] = (heat[i + 1, j] + heat[i - 1, j] + heat[i, j - 1] + tl + (L - 4) * heat[i, j]) / L

        for i in range(0, n - 1):
            for j in range(0, n - 1):
                heat[i, j] = (heat[i+1, j] + heat[i-1, j] + heat[i, j-1] + heat[i, j+1] + (L - 4) * heat[i, j]) / L

        t = t + dt

    return heat


def uniform_dist(low,
                 high,
                 input_size=10 ** 6,
                 output_size=10 ** 4,
                 random=False
                 ):
    """
    Generates and select data from uniform distributions.

    Args:
        low (float): Minimum value of the sample to be generated.
        high (float): Maximum value of the sample to be generated.
        input_size (int): Size of the sample in the uniform distribution.
        output_size (int): Size of the sample to return.
        random (bool): Whether or not to apply random choice from the sample.

    Return:
        Return generated data from uniform distributions of a given interval.
        Note that replacement is used when input_size < output_size.

    """

    import numpy as np

    # Generate a uniform distribution
    uniform = np.random.uniform(low=low, high=high, size=input_size)
    uniform = np.round(uniform, 3)

    if not random:
        data = uniform

    # Select and return a desired data sample.
    else:
        rep = False if input_size >= output_size else True
        data = np.random.choice(uniform, size=output_size, replace=rep)
    return data
