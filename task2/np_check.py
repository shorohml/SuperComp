import numpy as np
from scipy import integrate


def f(x, y, z):
    xz_sq = x*x + z*z
    if xz_sq + y * y <= 1:
        return np.sin(xz_sq) * y
    else:
        return 0.0


if __name__ == '__main__':
    options = {'limit': 300}
    y, abserr = integrate.nquad(
        f,
        [[-1, 1], [-1, 1], [-1, 1]],
        opts=[options, options, options]
    )
    print(y, abserr)
