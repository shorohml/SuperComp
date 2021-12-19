#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
// #define NDEBUG
#include <cassert>

// TODO: write proper argument parser
const double L_x = 1.0;
const double L_y = 1.0;
const double L_z = 1.0;
const double T = 1.0;
const size_t N = 128;
const size_t K = 400;
const bool SAVE_LAYERS = true;
const size_t SAVE_STEPS = 100;
const size_t P_x = 10;
const size_t P_y = 10;
const size_t P_z = 10;

double u(double x, double y, double z, double a_t, double t) {
    return sin(2 * M_PI * x / L_x) * sin(4 * M_PI * y / L_y) * sin(6 * M_PI * z / L_z) *
           cos(a_t * t);
}

template <typename T> void save_layer(std::vector<T> &layer, std::string path) {
    std::ofstream fs(path, std::ios::out | std::ios::binary);
    for (const double &val : layer) {
        fs.write((char *)&val, sizeof(double));
    }
    fs.close();
}

template <typename T> class Grid3D {
  private:
    std::vector<T> data;
    std::size_t P_x;
    std::size_t P_y;
    std::size_t P_z;
    std::size_t size;

  public:
    Grid3D() : P_x(0), P_y(0), P_z(0), size(0) {}

    Grid3D(std::size_t _P_x, std::size_t _P_y, std::size_t _P_z) {
        P_x = _P_x;
        P_y = _P_y;
        P_z = _P_z;
        size = P_x * P_y * P_z;
        data = std::vector<T>(size);
    }

    T &get(std::size_t i, std::size_t j, std::size_t k) { return data[(i * P_y + j) * P_z + k]; }

    void set(std::size_t i, std::size_t j, std::size_t k, const T &val) {
        data[(i * P_y + j) * P_z + k] = val;
    }

    std::size_t get_P_x() { return P_x; }

    std::size_t get_P_y() { return P_y; }

    std::size_t get_P_z() { return P_z; }
};

template <typename T> struct Block3D {
  public:
    Grid3D<T> grid;

    std::vector<int> start;
    std::vector<int> finish;
    std::vector<int> dims;
    int size;

    Block3D() { Block3D(0, 0, 0, 0, 0, 0); }

    Block3D(int start_i, int start_j, int start_k, int finish_i, int finish_j, int finish_k) {
        assert(finish_i >= start_i);
        assert(finish_j >= start_j);
        assert(finish_k >= start_k);

        start = {start_i, start_j, start_k};
        finish = {finish_i, finish_j, finish_k};
        dims = {finish_i - start_i, finish_j - start_j, finish_k - start_k};
        size = dims[0] * dims[1] * dims[2];
        grid = Grid3D<T>(dims[0], dims[1], dims[2]);
    }
};

// class GridBlockSeparator {
//   private:
//     int P_x = 1, P_y = 1, P_z = 1;
//     std::vector<int> factorization;
//     std::vector<std::vector<int>> factorizations;

//     void add_factorization(int num, int level);

//   public:
//     void compute_factorizations(int num, int num_factors);

//     void separate(int num_blocks);

//     std::vector<std::vector<int>> get_factorizations() { return factorizations; }

//     int get_P_x() { return P_x; }

//     int get_P_y() { return P_y; }

//     int get_P_z() { return P_z; }
// };

// void GridBlockSeparator::add_factorization(int num, int level) {
//     if (level == 0) {
//         factorization.push_back(num);
//         factorizations.push_back(factorization);
//         factorization.pop_back();
//         return;
//     }

//     int start = factorization.size() == 0 ? 1 : factorization[factorization.size() - 1];
//     int next;
//     for (int i = start; i < num + 1; ++i) {
//         if (0 == num % i && (next = num / i) >= i) {
//             factorization.push_back(i);
//             add_factorization(next, level - 1);
//             factorization.pop_back();
//         }
//     }
// }

// void GridBlockSeparator::compute_factorizations(int num, int num_factors) {
//     factorization = std::vector<int>();
//     factorizations = std::vector<std::vector<int>>();
//     add_factorization(num, num_factors - 1);
// }

// void GridBlockSeparator::separate(int num_blocks) {
//     compute_factorizations(num_blocks, 3);
//     if (1 == factorizations.size()) {
//         P_x = 1;
//         P_y = 1;
//         P_z = num_blocks;
//     } else if (2 == factorizations.size()) {
//         P_x = 1;
//         P_y = factorizations[1][1];
//         P_z = factorizations[1][2];
//     } else {
//         // minimize P_x + P_y + P_z
//         int sum = factorizations[0][0] + factorizations[0][1] + factorizations[0][2];
//         int min_idx = 0, min = sum;
//         for (int i = 1; i < factorizations.size(); ++i) {
//             sum = factorizations[i][0] + factorizations[i][1] + factorizations[i][2];
//             if (sum < min) {
//                 min = sum;
//                 min_idx = i;
//             }
//         }
//         P_x = factorizations[min_idx][0];
//         P_y = factorizations[min_idx][1];
//         P_z = factorizations[min_idx][2];
//     }
// }

template <typename T> struct Block3DBound {
  public:
    std::vector<int> dims;
    std::vector<std::vector<T>> faces;

    Block3DBound(int dim_0, int dim_1, int dim_2) {
        assert(dim_0 >= 1);
        assert(dim_1 >= 1);
        assert(dim_2 >= 1);

        dims = {dim_0, dim_1, dim_2};
        faces = std::vector<std::vector<T>>(6);
        faces[0].resize(dim_1 * dim_2);
        faces[1].resize(dim_1 * dim_2);
        faces[2].resize(dim_0 * dim_2);
        faces[3].resize(dim_0 * dim_2);
        faces[4].resize(dim_0 * dim_1);
        faces[5].resize(dim_0 * dim_1);
    }
};

enum AXIS { X = 0, Y, Z };

template <typename T>
void pack_face(Block3D<T> block, AXIS axis, bool first, std::vector<T> &face) {
    std::size_t face_size;
    switch (axis) {
    case AXIS::X:
        face_size = block.dims[1] * block.dims[2];
        break;
    case AXIS::Y:
        face_size = block.dims[0] * block.dims[2];
        break;
    case AXIS::Z:
        face_size = block.dims[0] * block.dims[1];
        break;
    }
    face.resize(face_size);
    int face_idx = first ? 0 : block.dims[axis] - 1;

    switch (axis) {
    case AXIS::X:
#pragma omp parallel for
        for (int j = 0; j < block.dims[1]; ++j) {
            for (int k = 0; k < block.dims[2]; ++k) {
                face[j * block.dims[2] + k] = block.grid.get(face_idx, j, k);
            }
        }
        break;
    case AXIS::Y:
#pragma omp parallel for
        for (int i = 0; i < block.dims[0]; ++i) {
            for (int k = 0; k < block.dims[2]; ++k) {
                face[i * block.dims[2] + k] = block.grid.get(i, face_idx, k);
            }
        }
        break;
    case AXIS::Z:
#pragma omp parallel for
        for (int i = 0; i < block.dims[0]; ++i) {
            for (int j = 0; j < block.dims[1]; ++j) {
                face[i * block.dims[1] + j] = block.grid.get(i, j, face_idx);
            }
        }
        break;
    }
}

template <typename T> T find_value(Block3D<T> &block, Block3DBound<T> &bound, int i, int j, int k) {
    if (i < 0) {
        return bound.faces[0][j * block.dims[2] + k];
    } else if (i >= block.dims[0]) {
        return bound.faces[1][j * block.dims[2] + k];
    } else if (j < 0) {
        return bound.faces[2][i * block.dims[2] + k];
    } else if (j >= block.dims[1]) {
        return bound.faces[3][i * block.dims[2] + k];
    } else if (k < 0) {
        return bound.faces[4][i * block.dims[1] + j];
    } else if (k >= block.dims[2]) {
        return bound.faces[5][i * block.dims[1] + j];
    }
    return block.grid.get(i, j, k);
}

int main(int argc, char **argv) {
    double a_t = M_PI * sqrt(4 / (L_x * L_x) + 16 / (L_y * L_y) + 36 / (L_z * L_z));
    double delta, max_err;
    const double tau = T / K;
    const double h_x = L_x / N;
    const double h_y = L_x / N;
    const double h_z = L_x / N;
    int size, rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[3] = {0, 0, 0};

    MPI_Dims_create(size, 3, dims);

    if (0 == rank) {
        std::cout << std::endl << dims[0] << ' ' << dims[1] << ' ' << dims[2] << std::endl;
    }

    int periods[3] = {true, true, true};

    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &grid_comm);

    int coords[3];
    MPI_Cart_coords(grid_comm, rank, 3, coords);

    std::cout << std::endl << coords[0] << ' ' << coords[1] << ' ' << coords[2] << std::endl;

    int block_size[3];
    for (int i = 0; i < 3; ++i) {
        block_size[i] = (N + 1) / dims[i];
    }
    int rem[3];
    for (int i = 0; i < 3; ++i) {
        rem[i] = (N + 1) % dims[i];
    }
    int smaller_block_start[3];
    for (int i = 0; i < 3; ++i) {
        smaller_block_start[i] = rem[i] * (block_size[i] + 1);
    }

    int block_start[3];
    int block_finish[3];
    for (int i = 0; i < 3; ++i) {
        if (coords[i] < rem[i]) {
            block_start[i] = coords[i] * (block_size[i] + 1);
            block_finish[i] = block_start[i] + block_size[i] + 1;
        } else {
            block_start[i] = smaller_block_start[i] + (coords[i] - rem[i]) * block_size[i];
            block_finish[i] = block_start[i] + block_size[i];
        }
    }

    Block3D<double> block_0(block_start[0], block_start[1], block_start[2], block_finish[0],
                            block_finish[1], block_finish[2]);
    Block3D<double> block_1(block_start[0], block_start[1], block_start[2], block_finish[0],
                            block_finish[1], block_finish[2]);
    Block3D<double> block_2(block_start[0], block_start[1], block_start[2], block_finish[0],
                            block_finish[1], block_finish[2]);
    Block3D<double> errs(block_start[0], block_start[1], block_start[2], block_finish[0],
                         block_finish[1], block_finish[2]);
    Block3D<double> tmp_block;

#pragma omp parallel for collapse(3)
    for (int i = block_0.start[0]; i < block_0.finish[0]; ++i) {
        for (int j = block_0.start[1]; j < block_0.finish[1]; ++j) {
            for (int k = block_0.start[2]; k < block_0.finish[2]; ++k) {
                block_0.grid.set(i - block_0.start[0], j - block_0.start[1], k - block_0.start[2],
                                 u(L_x * i / N, L_y * j / N, L_z * k / N, a_t, 0.0));
            }
        }
    }

    Block3DBound<double> sendbound(block_0.dims[0], block_0.dims[1], block_0.dims[2]);
    Block3DBound<double> recvbound(block_0.dims[0], block_0.dims[1], block_0.dims[2]);

    // TODO: remove unecessary sends on grid boundary
    int src, dst;
    for (int axis = 0; axis < 3; ++axis) {
        MPI_Cart_shift(grid_comm, axis, 1, &src, &dst);

        pack_face(block_0, static_cast<AXIS>(axis), true, sendbound.faces[2 * axis]);
        MPI_Sendrecv(sendbound.faces[2 * axis].data(), sendbound.faces[2 * axis].size(), MPI_DOUBLE,
                     src, 0, recvbound.faces[2 * axis + 1].data(),
                     recvbound.faces[2 * axis + 1].size(), MPI_DOUBLE, dst, 0, grid_comm,
                     MPI_STATUS_IGNORE);

        pack_face(block_0, static_cast<AXIS>(axis), false, sendbound.faces[2 * axis + 1]);
        MPI_Sendrecv(sendbound.faces[2 * axis + 1].data(), sendbound.faces[2 * axis + 1].size(),
                     MPI_DOUBLE, dst, 0, recvbound.faces[2 * axis].data(),
                     recvbound.faces[2 * axis].size(), MPI_DOUBLE, src, 0, grid_comm,
                     MPI_STATUS_IGNORE);
    }

#pragma omp parallel for collapse(3)
    for (int i = 0; i < block_1.dims[0]; ++i) {
        for (int j = 0; j < block_1.dims[1]; ++j) {
            for (int k = 0; k < block_1.dims[2]; ++k) {
                if ((0 == coords[0] && 0 == i) || (0 == coords[1] && 0 == j) ||
                    (0 == coords[2] && 0 == k) ||
                    (dims[0] - 1 == coords[0] && block_1.dims[0] - 1 == i) ||
                    (dims[1] - 1 == coords[1] && block_1.dims[1] - 1 == j) ||
                    (dims[2] - 1 == coords[2] && block_1.dims[2] - 1 == k)) {
                    block_1.grid.set(i, j, k,
                                     u(L_x * (i + block_1.start[0]) / N,
                                       L_y * (j + block_1.start[1]) / N,
                                       L_z * (k + block_1.start[2]) / N, a_t, tau));
                } else {
                    delta = 0.0;
                    delta += (find_value(block_0, recvbound, i - 1, j, k) -
                              2 * block_0.grid.get(i, j, k) +
                              find_value(block_0, recvbound, i + 1, j, k)) /
                             (h_x * h_x);
                    delta += (find_value(block_0, recvbound, i, j - 1, k) -
                              2 * block_0.grid.get(i, j, k) +
                              find_value(block_0, recvbound, i, j + 1, k)) /
                             (h_y * h_y);
                    delta += (find_value(block_0, recvbound, i, j, k - 1) -
                              2 * block_0.grid.get(i, j, k) +
                              find_value(block_0, recvbound, i, j, k + 1)) /
                             (h_z * h_z);
                    block_1.grid.set(i, j, k,
                                     block_0.grid.get(i, j, k) + (tau * tau) / 2.0 * delta);
                }
            }
        }
    }

    // compute approximation
    for (size_t t_step = 2; t_step < K + 1; ++t_step) {
        for (int axis = 0; axis < 3; ++axis) {
            MPI_Cart_shift(grid_comm, axis, 1, &src, &dst);

            pack_face(block_1, static_cast<AXIS>(axis), true, sendbound.faces[2 * axis]);
            MPI_Sendrecv(sendbound.faces[2 * axis].data(), sendbound.faces[2 * axis].size(),
                         MPI_DOUBLE, src, 0, recvbound.faces[2 * axis + 1].data(),
                         recvbound.faces[2 * axis + 1].size(), MPI_DOUBLE, dst, 0, grid_comm,
                         MPI_STATUS_IGNORE);

            pack_face(block_1, static_cast<AXIS>(axis), false, sendbound.faces[2 * axis + 1]);
            MPI_Sendrecv(sendbound.faces[2 * axis + 1].data(), sendbound.faces[2 * axis + 1].size(),
                         MPI_DOUBLE, dst, 0, recvbound.faces[2 * axis].data(),
                         recvbound.faces[2 * axis].size(), MPI_DOUBLE, src, 0, grid_comm,
                         MPI_STATUS_IGNORE);
        }

#pragma omp parallel for collapse(3)
        for (int i = 0; i < block_2.dims[0]; ++i) {
            for (int j = 0; j < block_2.dims[1]; ++j) {
                for (int k = 0; k < block_2.dims[2]; ++k) {
                    if ((0 == coords[0] && 0 == i) || (0 == coords[1] && 0 == j) ||
                        (0 == coords[2] && 0 == k) ||
                        (dims[0] - 1 == coords[0] && block_1.dims[0] - 1 == i) ||
                        (dims[1] - 1 == coords[1] && block_1.dims[1] - 1 == j) ||
                        (dims[2] - 1 == coords[2] && block_1.dims[2] - 1 == k)) {
                        block_2.grid.set(i, j, k,
                                         u(L_x * (i + block_2.start[0]) / N,
                                           L_y * (j + block_2.start[1]) / N,
                                           L_z * (k + block_2.start[2]) / N, a_t, tau * t_step));
                    } else {
                        delta = 0.0;
                        delta += (find_value(block_1, recvbound, i - 1, j, k) -
                                  2 * block_1.grid.get(i, j, k) +
                                  find_value(block_1, recvbound, i + 1, j, k)) /
                                 (h_x * h_x);
                        delta += (find_value(block_1, recvbound, i, j - 1, k) -
                                  2 * block_1.grid.get(i, j, k) +
                                  find_value(block_1, recvbound, i, j + 1, k)) /
                                 (h_y * h_y);
                        delta += (find_value(block_1, recvbound, i, j, k - 1) -
                                  2 * block_1.grid.get(i, j, k) +
                                  find_value(block_1, recvbound, i, j, k + 1)) /
                                 (h_z * h_z);
                        block_2.grid.set(i, j, k,
                                         2 * block_1.grid.get(i, j, k) - block_0.grid.get(i, j, k) +
                                             (tau * tau) * delta);
                    }
                }
            }
        }

        max_err = 0.0;
        double err;
        for (int i = 0; i < block_2.dims[0]; ++i) {
            for (int j = 0; j < block_2.dims[1]; ++j) {
                for (int k = 0; k < block_2.dims[2]; ++k) {
                    err = fabs(block_2.grid.get(i, j, k) -
                               u(L_x * (i + block_2.start[0]) / N, L_y * (j + block_2.start[1]) / N,
                                 L_z * (k + block_2.start[2]) / N, a_t, tau * t_step));
                    errs.grid.set(i, j, k, err);
                    if (err > max_err) {
                        max_err = err;
                    }
                }
            }
        }

        double global_max;
        MPI_Reduce(&max_err, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
        if (0 == rank) {
            std::cout << "layer " << t_step << " error: " << global_max << std::endl;
        }

        tmp_block = std::move(block_0);
        block_0 = std::move(block_1);
        block_1 = std::move(block_2);
        block_2 = std::move(tmp_block);
    }
    MPI_Finalize();
    return 0;
}