#include "Solver.h"
#include <iostream>
#include <mpi.h>

void Solver::compute_block_coords() {
    int rem[3];
    int smaller_block_start[3];

    // (config.N[i] + 1) % dims[i] blocks with size block_size[i] + 1
    // other blocks with size block_size[i]
    for (int i = 0; i < 3; ++i) {
        block_size[i] = (config.N[i] + 1) / dims[i];
        rem[i] = (config.N[i] + 1) % dims[i];
        smaller_block_start[i] = rem[i] * (block_size[i] + 1);

        if (coords[i] < rem[i]) {
            block_start[i] = coords[i] * (block_size[i] + 1);
            block_finish[i] = block_start[i] + block_size[i] + 1;
        } else {
            block_start[i] = smaller_block_start[i] + (coords[i] - rem[i]) * block_size[i];
            block_finish[i] = block_start[i] + block_size[i];
        }
        block_size[i] = block_finish[i] - block_start[i];
    }
}

Solver::Solver(const std::string &config_path, int argc, char **argv)
    : config(config_path), u4d(config.L[0], config.L[1], config.L[2]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start = MPI_Wtime();

    if (0 == rank) {
        config.print();
    }

    tau = config.T / config.K;
    tau_sq = tau * tau;
    tau_sq_half = tau_sq / 2.0;
    for (int i = 0; i < 3; ++i) {
        h[i] = config.L[0] / config.N[i];
        L_N[i] = config.L[i] / config.N[i];
        h_inv_sq[i] = 1.0 / (h[i] * h[i]);
    }

    // create cart communicator
    dims[0] = 0;
    dims[1] = 0;
    dims[2] = 0;
    periods[0] = true;
    periods[1] = true;
    periods[2] = true;
    MPI_Dims_create(size, 3, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &comm);
    MPI_Cart_coords(comm, rank, 3, coords);

    compute_block_coords();
    for (int i = 0; i < 3; ++i) {
        blocks[i].resize(block_start[0], block_start[1], block_start[2], block_finish[0],
                         block_finish[1], block_finish[2]);
    }
    bound.resize(block_size[0], block_size[1], block_size[2]);
    if (config.save_layers) {
        // only need to save those blocks in memory to later save to disk
        errs.resize(block_start[0], block_start[1], block_start[2], block_finish[0],
                    block_finish[1], block_finish[2]);
        analytical.resize(block_start[0], block_start[1], block_start[2], block_finish[0],
                          block_finish[1], block_finish[2]);
    }

    // create types
    MPI_Type_vector(block_size[0], block_size[2], block_size[1] * block_size[2], MPI_DOUBLE,
                    &face_type_1);
    MPI_Type_commit(&face_type_1);
    MPI_Type_vector(block_size[0] * block_size[1], 1, block_size[2], MPI_DOUBLE, &face_type_2);
    MPI_Type_commit(&face_type_2);
}

void Solver::compute_layer_0(Block3D<double> &block) {
#pragma omp parallel for
    for (int i = 0; i < block.dims[0]; ++i) {
        for (int j = 0; j < block.dims[1]; ++j) {
            for (int k = 0; k < block.dims[2]; ++k) {
                block.grid(i, j, k) =
                    u4d(L_N[0] * (i + block.start[0]), L_N[1] * (j + block.start[1]),
                        L_N[2] * (k + block.start[2]), 0.0);
            }
        }
    }
}

void Solver::send_data(Block3D<double> &block) {
    int src, dst;

    // axis 0
    MPI_Cart_shift(comm, 0, 1, &src, &dst);
    MPI_Sendrecv(block.grid.data(), bound.faces[1].size(), MPI_DOUBLE, src, 0,
                 bound.faces[1].data(), bound.faces[1].size(), MPI_DOUBLE, dst, 0, comm,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(block.grid.data() + (block.dims[0] - 1) * bound.faces[1].size(),
                 bound.faces[1].size(), MPI_DOUBLE, dst, 0, bound.faces[0].data(),
                 bound.faces[0].size(), MPI_DOUBLE, src, 0, comm, MPI_STATUS_IGNORE);

    // axis 1
    MPI_Cart_shift(comm, 1, 1, &src, &dst);
    MPI_Sendrecv(block.grid.data(), 1, face_type_1, src, 0, bound.faces[3].data(),
                 bound.faces[3].size(), MPI_DOUBLE, dst, 0, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(block.grid.data() + (block.dims[1] - 1) * block.dims[2], 1, face_type_1, dst, 0,
                 bound.faces[2].data(), bound.faces[2].size(), MPI_DOUBLE, src, 0, comm,
                 MPI_STATUS_IGNORE);

    // axis 2
    MPI_Cart_shift(comm, 2, 1, &src, &dst);
    MPI_Sendrecv(block.grid.data(), 1, face_type_2, src, 0, bound.faces[5].data(),
                 bound.faces[5].size(), MPI_DOUBLE, dst, 0, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(block.grid.data() + block.dims[2] - 1, 1, face_type_2, dst, 0,
                 bound.faces[4].data(), bound.faces[4].size(), MPI_DOUBLE, src, 0, comm,
                 MPI_STATUS_IGNORE);
}

double Solver::find_value(Block3D<double> &block, Block3DBound<double> &bound, int i, int j,
                          int k) {
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
    return block.grid(i, j, k);
}

double Solver::laplacian(Block3D<double> &block, int i, int j, int k) {
    double delta = 0.0;
    delta += (find_value(block, bound, i - 1, j, k) - 2 * block.grid(i, j, k) +
              find_value(block, bound, i + 1, j, k)) *
             h_inv_sq[0];
    delta += (find_value(block, bound, i, j - 1, k) - 2 * block.grid(i, j, k) +
              find_value(block, bound, i, j + 1, k)) *
             h_inv_sq[1];
    delta += (find_value(block, bound, i, j, k - 1) - 2 * block.grid(i, j, k) +
              find_value(block, bound, i, j, k + 1)) *
             h_inv_sq[2];
    return delta;
}

void Solver::compute_layer_1(Block3D<double> &block_0, Block3D<double> &block_1) {
#pragma omp parallel for
    for (int i = 0; i < block_1.dims[0]; ++i) {
        for (int j = 0; j < block_1.dims[1]; ++j) {
            for (int k = 0; k < block_1.dims[2]; ++k) {
                if ((0 == coords[0] && 0 == i) || (0 == coords[1] && 0 == j) ||
                    (0 == coords[2] && 0 == k) ||
                    (dims[0] - 1 == coords[0] && block_1.dims[0] - 1 == i) ||
                    (dims[1] - 1 == coords[1] && block_1.dims[1] - 1 == j) ||
                    (dims[2] - 1 == coords[2] && block_1.dims[2] - 1 == k)) {
                    block_1.grid(i, j, k) =
                        u4d(L_N[0] * (i + block_1.start[0]), L_N[1] * (j + block_1.start[1]),
                            L_N[2] * (k + block_1.start[2]), tau);
                } else {
                    block_1.grid(i, j, k) =
                        block_0.grid(i, j, k) + tau_sq_half * laplacian(block_0, i, j, k);
                }
            }
        }
    }
}

void Solver::compute_layer_2(Block3D<double> &block_0, Block3D<double> &block_1,
                             Block3D<double> &block_2) {
#pragma omp parallel for
    for (int i = 0; i < block_2.dims[0]; ++i) {
        for (int j = 0; j < block_2.dims[1]; ++j) {
            for (int k = 0; k < block_2.dims[2]; ++k) {
                if ((0 == coords[0] && 0 == i) || (0 == coords[1] && 0 == j) ||
                    (0 == coords[2] && 0 == k) ||
                    (dims[0] - 1 == coords[0] && block_2.dims[0] - 1 == i) ||
                    (dims[1] - 1 == coords[1] && block_2.dims[1] - 1 == j) ||
                    (dims[2] - 1 == coords[2] && block_2.dims[2] - 1 == k)) {
                    block_1.grid(i, j, k) =
                        u4d(L_N[0] * (i + block_1.start[0]), L_N[1] * (j + block_1.start[1]),
                            L_N[2] * (k + block_1.start[2]), tau);
                } else {
                    block_2.grid(i, j, k) = 2 * block_1.grid(i, j, k) - block_0.grid(i, j, k) +
                                            tau_sq * laplacian(block_1, i, j, k);
                }
            }
        }
    }
}

double Solver::compute_max_err(Block3D<double> &block, double t) {
    double err, val, max_err = 0.0;
    for (int i = 0; i < block.dims[0]; ++i) {
        for (int j = 0; j < block.dims[1]; ++j) {
            for (int k = 0; k < block.dims[2]; ++k) {
                val = u4d(L_N[0] * (i + block.start[0]), L_N[1] * (j + block.start[1]),
                          L_N[2] * (k + block.start[2]), tau * t);
                err = fabs(block.grid(i, j, k) - val);
                if (config.save_layers) {
                    analytical.grid(i, j, k) = val;
                    errs.grid(i, j, k) = err;
                }
                if (err > max_err) {
                    max_err = err;
                }
            }
        }
    }
    return max_err;
}

void Solver::save_layer(Block3D<double> &block, const std::string &path) {
    MPI_Request request;
    MPI_Status status;

    if (0 != rank) {
        MPI_Isend(block.grid.data(), block.grid.size(), MPI_DOUBLE, 0, 0, comm, &request);
        MPI_Waitall(1, &request, &status);
    } else {
        Grid3D<double> grid(config.N[0] + 1, config.N[1] + 1, config.N[2] + 1);

        // collect grid from blocks
        for (int i = block.start[0]; i < block.finish[0]; ++i) {
            for (int j = block.start[1]; j < block.finish[1]; ++j) {
                for (int k = block.start[2]; k < block.finish[2]; ++k) {
                    grid(i, j, k) =
                        block.grid(i - block.start[0], j - block.start[1], k - block.start[2]);
                }
            }
        }
        for (int send_rank = 1; send_rank < size; ++send_rank) {
            MPI_Cart_coords(comm, send_rank, 3, coords);
            compute_block_coords();
            Block3D<double> send_block(block_start[0], block_start[1], block_start[2],
                                       block_finish[0], block_finish[1], block_finish[2]);

            MPI_Irecv(send_block.grid.data(), send_block.grid.size(), MPI_DOUBLE, send_rank, 0,
                      comm, &request);
            MPI_Waitall(1, &request, &status);

            for (int i = send_block.start[0]; i < send_block.finish[0]; ++i) {
                for (int j = send_block.start[1]; j < send_block.finish[1]; ++j) {
                    for (int k = send_block.start[2]; k < send_block.finish[2]; ++k) {
                        grid(i, j, k) =
                            send_block.grid(i - send_block.start[0], j - send_block.start[1],
                                            k - send_block.start[2]);
                    }
                }
            }
        }
        // restore coords
        MPI_Cart_coords(comm, 0, 3, coords);
        compute_block_coords();

        // save cllected grid
        save_grid(grid, path);
    }
}

void Solver::run() {
    double block_max_err, layer_max_err, max_err;

    // layer 0
    compute_layer_0(blocks[0]);

    // layer 1
    send_data(blocks[0]);
    compute_layer_1(blocks[0], blocks[1]);

    block_max_err = compute_max_err(blocks[1], 1);
    MPI_Reduce(&block_max_err, &max_err, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (0 == rank) {
        std::cout << "Layer 1 error: " << max_err << std::endl;
    }

    // layers 2...K
    for (int t = 2; t < config.K + 1; ++t) {
        send_data(blocks[1]);
        compute_layer_2(blocks[0], blocks[1], blocks[2]);

        // compute err
        block_max_err = compute_max_err(blocks[2], t);
        MPI_Reduce(&block_max_err, &layer_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (0 == rank) {
            std::cout << "Layer " << t << " error: " << layer_max_err << std::endl;
        }
        if (layer_max_err > max_err) {
            max_err = layer_max_err;
        }

        // save layer to disk
        if (config.save_layers && 0 == t % config.save_step) {
            char N_0_c_str[50], N_1_c_str[50], N_2_c_str[50], t_c_str[50];
            sprintf(N_0_c_str, "%d", config.N[0] + 1);
            sprintf(N_1_c_str, "%d", config.N[1] + 1);
            sprintf(N_2_c_str, "%d", config.N[2] + 1);
            sprintf(t_c_str, "%d", t);

            std::string t_str(t_c_str);
            std::string grid_dim_str = std::string(N_0_c_str) + "_" + std::string(N_1_c_str) + "_" +
                                       std::string(N_2_c_str);

            save_layer(blocks[2],
                       config.layers_path + "layer_" + t_str + "_" + grid_dim_str + ".bin");
            save_layer(errs, config.layers_path + "errs_" + t_str + "_" + grid_dim_str + ".bin");
            save_layer(analytical,
                       config.layers_path + "analytical_" + t_str + "_" + grid_dim_str + ".bin");
        }

        // change blocks (2 -> 1, 1 -> 0)
        std::swap(blocks[0], blocks[2]);
        std::swap(blocks[0], blocks[1]);
    }
    if (0 == rank) {
        std::cout << std::endl << "Maximum error: " << max_err << std::endl;
    }
}

Solver::~Solver() {
    double ellapsed, max_ellapsed;

    finish = MPI_Wtime();

    ellapsed = finish - start;
    MPI_Reduce(&ellapsed, &max_ellapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank) {
        std::cout << "Ellapsed time: " << max_ellapsed << std::endl;
    }

    MPI_Finalize();
}
