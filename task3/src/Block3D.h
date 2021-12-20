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
        start[0] = start_i;
        start[1] = start_j;
        start[2] = start_k;
        finish[0] = finish_i;
        finish[1] = finish_j;
        finish[2] = finish_k;
        dims[0] = finish_i - start_i;
        dims[1] = finish_j - start_j;
        dims[2] = finish_k - start_k;
        grid = Grid3D<T>(dims[0], dims[1], dims[2]);
    }
};

template <typename T> struct Block3DBound {
  public:
    int dims[3];
    std::vector<std::vector<T>> faces;

    Block3DBound(int dim_0, int dim_1, int dim_2) {
        dims[0] = dim_0;
        dims[1] = dim_1;
        dims[2] = dim_2;
        faces = std::vector<std::vector<T>>(6);
        faces[0].resize(dim_1 * dim_2);
        faces[1].resize(dim_1 * dim_2);
        faces[2].resize(dim_0 * dim_2);
        faces[3].resize(dim_0 * dim_2);
        faces[4].resize(dim_0 * dim_1);
        faces[5].resize(dim_0 * dim_1);
    }
};