#pragma once

#include <cmath>

template <typename T> class U4D {
  private:
    T L_x, L_y, L_z, a_t;

  public:
    U4D(T _L_x, T _L_y, T _L_z) : L_x(_L_x), L_y(_L_y), L_z(_L_z) {
        a_t = M_PI * sqrt(4 / (L_x * L_x) + 16 / (L_y * L_y) + 36 / (L_z * L_z));
    }

    T operator()(T x, T y, T z, T t) const {
        return sin(2 * M_PI * x / L_x) * sin(4 * M_PI * y / L_y) * sin(6 * M_PI * z / L_z) *
               cos(a_t * t);
    }
};
