#pragma once

#include "Block3D.h"
#include "Config.h"
#include "U4D.h"
#include <mpi.h>

class Solver {
  private:
    Config config;

    // process and block coordinates
    int size, rank;
    int dims[3] = {0, 0, 0};
    int periods[3] = {true, true, true};
    int coords[3];
    MPI_Comm comm;

    // grid blocks
    int block_size[3];
    int block_start[3];
    int block_finish[3];
    Block3D<double> blocks[3];
    Block3D<double> errs;
    Block3D<double> analytical;
    Block3DBound<double> bound;

    // types for sends
    MPI_Datatype face_type_1;
    MPI_Datatype face_type_2;

    // u from hyperbolic equation
    U4D<double> u4d;

    // parameters
    double tau, tau_sq, tau_sq_half;
    double h[3], h_inv_sq[3];
    double L_N[3];
    double start, finish;

    void compute_block_coords();

    void compute_layer_0(Block3D<double> &block);

    void send_data(Block3D<double> &block);

    double find_value(Block3D<double> &block, Block3DBound<double> &bound, int i, int j, int k);

    double laplacian(Block3D<double> &block, int i, int j, int k);

    void compute_layer_1(Block3D<double> &block_0, Block3D<double> &block_1);

    void compute_layer_2(Block3D<double> &block_0, Block3D<double> &block_1,
                         Block3D<double> &block_2);

    double compute_max_err(Block3D<double> &block, double t);

    void save_layer(Block3D<double> &block, const std::string &path);

  public:
    Solver(const std::string &config_path, int argc, char **argv);

    void run();

    ~Solver();
};
