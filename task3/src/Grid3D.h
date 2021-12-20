#pragma once

#include <vector>
#include <fstream>
#include <string>

template <typename T> class Grid3D {
  private:
    std::vector<T> _buffer;
    int _P_x;
    int _P_y;
    int _P_z;
    int _size;

  public:
    Grid3D() : _P_x(0), _P_y(0), _P_z(0), _size(0) {}

    Grid3D(int P_x, int P_y, int P_z) {
        _P_x = P_x;
        _P_y = P_y;
        _P_z = P_z;
        _size = P_x * P_y * P_z;
        _buffer.resize(_size);
    }

    T *data() { return _buffer.data(); }

    T &operator()(int i, int j, int k) { return _buffer[(i * _P_y + j) * _P_z + k]; }

    int P_x() { return _P_x; }

    int P_y() { return _P_y; }

    int P_z() { return _P_z; }

    int size() { return _size; }
};

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
