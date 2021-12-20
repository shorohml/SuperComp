#include "Grid3D.h"

#include <fstream>

template <typename T> void Grid3D<T>::resize(int P_x, int P_y, int P_z) {
    _P_x = P_x;
    _P_y = P_y;
    _P_z = P_z;
    _size = P_x * P_y * P_z;
    _buffer.resize(_size);
}

void save_grid(Grid3D<double> &grid, const std::string &path) {
    std::ofstream fs(path.c_str(), std::ios::out | std::ios::binary);
    for (int i = 0; i < grid.P_x(); ++i) {
        for (int j = 0; j < grid.P_y(); ++j) {
            for (int k = 0; k < grid.P_z(); ++k) {
                fs.write((char *)&grid(i, j, k), sizeof(double));
            }
        }
    }
    fs.close();
}

template class Grid3D<float>;
template class Grid3D<double>;
