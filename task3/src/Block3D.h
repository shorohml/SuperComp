#pragma once

#include "Grid3D.h"

template <typename T> struct Block3D {
  public:
    Grid3D<T> grid;

    int start[3];
    int finish[3];
    int dims[3];

    Block3D() { Block3D(0, 0, 0, 0, 0, 0); }

    Block3D(int start_i, int start_j, int start_k, int finish_i, int finish_j, int finish_k) {
        resize(start_i, start_j, start_k, finish_i, finish_j, finish_k);
    }

    void resize(int start_i, int start_j, int start_k, int finish_i, int finish_j, int finish_k);
};

template <typename T> struct Block3DBound {
  public:
    int dims[3];
    std::vector<T> faces[6];

    Block3DBound() { Block3DBound(0, 0, 0); }

    Block3DBound(int dim_0, int dim_1, int dim_2) {
        resize(dim_0, dim_1, dim_2);
    }

    void resize(int dim_0, int dim_1, int dim_2);
};