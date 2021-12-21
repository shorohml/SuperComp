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
    int dims[3];
    int coords[3];
    bool is_first_block[3];
    bool is_last_block[3];
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

    void send_inner_values(Block3D<double> &block);

    void send_boundary_values(Block3D<double> &block);

    double find_value(Block3D<double> &block, Block3DBound<double> &bound, int i, int j, int k);

    double laplacian(Block3D<double> &block, int i, int j, int k);

    inline double get_boundary_val(Block3D<double> &block, int i, int j, int k, int t,
                            bool force_analytical, int axis);

    void compute_boundary_0(Block3D<double> &block, bool force_analytical, int i, int t);

    void compute_boundary_1(Block3D<double> &block, bool force_analytical, int j, int t);

    void compute_boundary_2(Block3D<double> &block, bool force_analytical, int k, int t);

    void compute_boundary(Block3D<double> &block, bool force_analytical, int t);

    void compute_layer_1(Block3D<double> &block_0, Block3D<double> &block_1);

    void compute_layer_2(Block3D<double> &block_0, Block3D<double> &block_1,
                         Block3D<double> &block_2, int t);

    double compute_max_err(Block3D<double> &block, double t);

    void save_layer(Block3D<double> &block, const std::string &path);

  public:
    Solver(const std::string &config_path, int argc, char **argv);

    void run();

    ~Solver();
};
