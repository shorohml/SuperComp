#include "Block3D.h"

template <typename T>
void Block3D<T>::resize(int start_i, int start_j, int start_k, int finish_i, int finish_j,
                        int finish_k) {
    start[0] = start_i;
    start[1] = start_j;
    start[2] = start_k;
    finish[0] = finish_i;
    finish[1] = finish_j;
    finish[2] = finish_k;
    dims[0] = finish_i - start_i;
    dims[1] = finish_j - start_j;
    dims[2] = finish_k - start_k;
    grid.resize(dims[0], dims[1], dims[2]);
}

template<typename T>
void Block3DBound<T>::resize(int dim_0, int dim_1, int dim_2) {
    dims[0] = dim_0;
    dims[1] = dim_1;
    dims[2] = dim_2;
    faces[0].resize(dim_1 * dim_2);
    faces[1].resize(dim_1 * dim_2);
    faces[2].resize(dim_0 * dim_2);
    faces[3].resize(dim_0 * dim_2);
    faces[4].resize(dim_0 * dim_1);
    faces[5].resize(dim_0 * dim_1);    
}

template class Block3D<float>;
template class Block3D<double>;
template class Block3DBound<float>;
template class Block3DBound<double>;
