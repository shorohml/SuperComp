#pragma once

#include <cmath>

template <typename T> class U4D {
  private:
    T mult[3];
    T a_t;

  public:
    U4D(T L_x, T L_y, T L_z) {
        mult[0] = 2 * M_PI / L_x;
        mult[1] = 4 * M_PI / L_y;
        mult[2] = 6 * M_PI / L_z;
        a_t = M_PI * sqrt(4 / (L_x * L_x) + 16 / (L_y * L_y) + 36 / (L_z * L_z));
    }

    T operator()(T x, T y, T z, T t) const {
        return sin(mult[0] * x) * sin(mult[1] * y) * sin(mult[2] * z) * cos(a_t * t);
    }
};
