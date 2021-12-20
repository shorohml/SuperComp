#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
// #define NDEBUG
#include <cassert>

#define BG

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

    T *get_data() { return data.data(); }

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

        start.push_back(start_i);
        start.push_back(start_j);
        start.push_back(start_k);
        finish.push_back(finish_i);
        finish.push_back(finish_j);
        finish.push_back(finish_k);
        dims.push_back(finish_i - start_i);
        dims.push_back(finish_j - start_j);
        dims.push_back(finish_k - start_k);
        size = dims[0] * dims[1] * dims[2];
        grid = Grid3D<T>(dims[0], dims[1], dims[2]);
    }
};

template <typename T> struct Block3DBound {
  public:
    std::vector<int> dims;
    std::vector<std::vector<T> > faces;

    Block3DBound(int dim_0, int dim_1, int dim_2) {
        assert(dim_0 >= 1);
        assert(dim_1 >= 1);
        assert(dim_2 >= 1);

        dims.push_back(dim_0);
        dims.push_back(dim_1);
        dims.push_back(dim_2);
        faces = std::vector<std::vector<T> >(6);
        faces[0].resize(dim_1 * dim_2);
        faces[1].resize(dim_1 * dim_2);
        faces[2].resize(dim_0 * dim_2);
        faces[3].resize(dim_0 * dim_2);
        faces[4].resize(dim_0 * dim_1);
        faces[5].resize(dim_0 * dim_1);
    }
};

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

#ifndef BG
void save_grid(Grid3D<double> &grid, const std::string &path) {
    std::ofstream fs(path, std::ios::out | std::ios::binary);
    for (std::size_t i = 0; i < grid.get_P_x(); ++i) {
        for (std::size_t j = 0; j < grid.get_P_y(); ++j) {
            for (std::size_t k = 0; k < grid.get_P_z(); ++k) {
                fs.write((char *)&grid.get(i, j, k), sizeof(double));
            }
        }
    }
    fs.close();
}

void save_layer(Block3D<double> &block, const std::string &path, int rank, int size, int grid_comm, int *dims) {
    MPI_Request request;
    MPI_Status status;

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

    if (0 != rank) {
        MPI_Isend(block.grid.get_data(), block.size, MPI_DOUBLE, 0, 0, grid_comm,
                    &request);
        MPI_Waitall(1, &request, &status);
    } else {
        Grid3D<double> grid(N + 1, N + 1, N + 1);

        for (int i = block.start[0]; i < block.finish[0]; ++i) {
            for (int j = block.start[1]; j < block.finish[1]; ++j) {
                for (int k = block.start[2]; k < block.finish[2]; ++k) {
                    grid.set(i, j, k,
                                block.grid.get(i - block.start[0], j - block.start[1],
                                                k - block.start[2]));
                }
            }
        }
        for (int send_rank = 1; send_rank < size; ++send_rank) {
            int send_coords[3];
            MPI_Cart_coords(grid_comm, send_rank, 3, send_coords);

            int block_start[3];
            int block_finish[3];
            for (int i = 0; i < 3; ++i) {
                if (send_coords[i] < rem[i]) {
                    block_start[i] = send_coords[i] * (block_size[i] + 1);
                    block_finish[i] = block_start[i] + block_size[i] + 1;
                } else {
                    block_start[i] =
                        smaller_block_start[i] + (send_coords[i] - rem[i]) * block_size[i];
                    block_finish[i] = block_start[i] + block_size[i];
                }
            }

            Block3D<double> send_block(block_start[0], block_start[1], block_start[2],
                                        block_finish[0], block_finish[1], block_finish[2]);

            MPI_Irecv(send_block.grid.get_data(), send_block.size, MPI_DOUBLE, send_rank, 0,
                        grid_comm, &request);
            MPI_Waitall(1, &request, &status);

            for (int i = send_block.start[0]; i < send_block.finish[0]; ++i) {
                for (int j = send_block.start[1]; j < send_block.finish[1]; ++j) {
                    for (int k = send_block.start[2]; k < send_block.finish[2]; ++k) {
                        grid.set(i, j, k,
                                    send_block.grid.get(i - send_block.start[0],
                                                        j - send_block.start[1],
                                                        k - send_block.start[2]));
                    }
                }
            }
        }

        save_grid(grid, path);
    }
}
#endif

int main(int argc, char **argv) {
    const double tau = T / K;
    const double h_x = L_x / N;
    const double h_y = L_x / N;
    const double h_z = L_x / N;
    const double a_t = M_PI * sqrt(4 / (L_x * L_x) + 16 / (L_y * L_y) + 36 / (L_z * L_z));
    int size, rank;
    double delta, max_err;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[3] = {0, 0, 0};

    MPI_Dims_create(size, 3, dims);

    int periods[3] = {true, true, true};

    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &grid_comm);

    int coords[3];
    MPI_Cart_coords(grid_comm, rank, 3, coords);

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

#pragma omp parallel for
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

    int src, dst;

    // axis 0
    MPI_Cart_shift(grid_comm, 0, 1, &src, &dst);
    MPI_Sendrecv(block_0.grid.get_data(), recvbound.faces[1].size(), MPI_DOUBLE, src, 0,
                 recvbound.faces[1].data(), recvbound.faces[1].size(), MPI_DOUBLE, dst, 0,
                 grid_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(block_0.grid.get_data() + (block_0.dims[0] - 1) * recvbound.faces[1].size(),
                 sendbound.faces[1].size(), MPI_DOUBLE, dst, 0, recvbound.faces[0].data(),
                 recvbound.faces[0].size(), MPI_DOUBLE, src, 0, grid_comm, MPI_STATUS_IGNORE);

    // axis 1
    MPI_Datatype face_type_1;
    MPI_Type_vector(block_0.dims[0], block_0.dims[2], block_0.dims[1] * block_0.dims[2], MPI_DOUBLE,
                    &face_type_1);
    MPI_Type_commit(&face_type_1);

    MPI_Cart_shift(grid_comm, 1, 1, &src, &dst);
    MPI_Sendrecv(block_0.grid.get_data(), 1, face_type_1, src, 0, recvbound.faces[3].data(),
                 recvbound.faces[3].size(), MPI_DOUBLE, dst, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(block_0.grid.get_data() + (block_0.dims[1] - 1) * block_0.dims[2], 1, face_type_1,
                 dst, 0, recvbound.faces[2].data(), recvbound.faces[2].size(), MPI_DOUBLE, src, 0,
                 grid_comm, MPI_STATUS_IGNORE);

    // axis 2
    MPI_Datatype face_type_2;
    MPI_Type_vector(block_0.dims[0] * block_0.dims[1], 1, block_0.dims[2], MPI_DOUBLE,
                    &face_type_2);
    MPI_Type_commit(&face_type_2);

    MPI_Cart_shift(grid_comm, 2, 1, &src, &dst);
    MPI_Sendrecv(block_0.grid.get_data(), 1, face_type_2, src, 0, recvbound.faces[5].data(),
                 recvbound.faces[5].size(), MPI_DOUBLE, dst, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(block_0.grid.get_data() + block_0.dims[2] - 1, 1, face_type_2, dst, 0,
                 recvbound.faces[4].data(), recvbound.faces[4].size(), MPI_DOUBLE, src, 0,
                 grid_comm, MPI_STATUS_IGNORE);

#pragma omp parallel for
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
        // axis 0
        MPI_Cart_shift(grid_comm, 0, 1, &src, &dst);
        MPI_Sendrecv(block_1.grid.get_data(), recvbound.faces[1].size(), MPI_DOUBLE, src, 0,
                     recvbound.faces[1].data(), recvbound.faces[1].size(), MPI_DOUBLE, dst, 0,
                     grid_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(block_1.grid.get_data() + (block_1.dims[0] - 1) * recvbound.faces[1].size(),
                     sendbound.faces[1].size(), MPI_DOUBLE, dst, 0, recvbound.faces[0].data(),
                     recvbound.faces[0].size(), MPI_DOUBLE, src, 0, grid_comm, MPI_STATUS_IGNORE);

        // axis 1
        MPI_Cart_shift(grid_comm, 1, 1, &src, &dst);
        MPI_Sendrecv(block_1.grid.get_data(), 1, face_type_1, src, 0, recvbound.faces[3].data(),
                     recvbound.faces[3].size(), MPI_DOUBLE, dst, 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(block_1.grid.get_data() + (block_1.dims[1] - 1) * block_1.dims[2], 1,
                     face_type_1, dst, 0, recvbound.faces[2].data(), recvbound.faces[2].size(),
                     MPI_DOUBLE, src, 0, grid_comm, MPI_STATUS_IGNORE);

        // axis 2
        MPI_Cart_shift(grid_comm, 2, 1, &src, &dst);
        MPI_Sendrecv(block_1.grid.get_data(), 1, face_type_2, src, 0, recvbound.faces[5].data(),
                     recvbound.faces[5].size(), MPI_DOUBLE, dst, 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(block_1.grid.get_data() + block_1.dims[2] - 1, 1, face_type_2, dst, 0,
                     recvbound.faces[4].data(), recvbound.faces[4].size(), MPI_DOUBLE, src, 0,
                     grid_comm, MPI_STATUS_IGNORE);

#pragma omp parallel for
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

        #ifndef BG
        if (0 == t_step % 100) {
            save_layer(block_2, "layer" + std::to_string(t_step) + ".bin", rank, size, grid_comm, dims);
            save_layer(errs, "errs" + std::to_string(t_step) + ".bin", rank, size, grid_comm, dims);
        }
        #endif

        std::swap(block_0, block_2);
        std::swap(block_0, block_1);
    }
    MPI_Finalize();
    return 0;
}