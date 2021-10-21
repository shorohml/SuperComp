import numpy as np
from scipy import integrate


def f(x, y, z):
    if x**2 + y**2 + z**2 <= 1:
        return np.sin(x**2 + z**2) * y
    else:
        return 0.0


if __name__ == '__main__':
    options = {'limit': 100}
    y, abserr = integrate.nquad(
        f,
        [[-1, 1], [-1, 1], [-1, 1]],
        opts=[options, options, options]
    )
    print(y, abserr)
