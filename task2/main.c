#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const double ANALITIC_I = 0.06225419868;
const double COORDS_MIN[3] = {0.0, 0.0, 0.0};
const double COORDS_MAX[3] = {1.0, 1.0, 1.0};
const int MAX_N_POINTS = 1000000; // max number of generated points at all
const int N_POINTS = 1;           // number of points generated in each MPI process in each step

double f(double x, double y, double z) {
    double xz_sq = x * x + z * z;
    if (xz_sq + y * y > 1) {
        return 0.0;
    } else {
        return sin(xz_sq) * y;
    }
}

int main(int argc, char **argv) {
    int size, rank;
    int count = 0, is_finished = 0;
    int step_n_points;
    double coords[3], coords_diff[3];
    double val, val_sum = 0.0;
    double volume, analytic_I = ANALITIC_I;
    double I, err, eps;
    double start, finish;
    double time, max_time;

    if (argc != 2) {
        printf("Usage: ./main {eps}\n");
        return 0;
    }
    eps = atof(argv[1]);
    if (eps <= 0.0) {
        printf("invalid eps\n");
        return 0;
    }

    MPI_Init(&argc, &argv);

    start = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // set random seed
    time = fmod(start, (double)INT_MAX - size);
    srand((int)time + rank);

    // compute domain volume
    if (rank == 0) {
        volume = 1.0;
        for (int i = 0; i < 3; ++i) {
            volume *= COORDS_MAX[i] - COORDS_MIN[i];
        }
        // for a small speedup
        analytic_I /= volume;
    }

    // precompute COORDS_MAX - COORDS_MAX and size * N_POINTS for a small speedup
    for (int i = 0; i < 3; ++i) {
        coords_diff[i] = COORDS_MAX[i] - COORDS_MIN[i];
    }
    step_n_points = size * N_POINTS;
    while (1) {
        val = 0.0;
        for (int i = 0; i < N_POINTS; ++i) {
            for (int j = 0; j < 3; ++j) {
                // in [0, 1]
                coords[j] = (double)rand() / INT_MAX;
                // in [COORDS_MIN, COORDS_MAX]
                coords[j] = coords[j] * coords_diff[j] + COORDS_MIN[j];
            }

            val += f(coords[0], coords[1], coords[2]);
        }
        if (rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&val, 0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        // check finishing criterion
        if (rank == 0) {
            val_sum += val;
            count += step_n_points;
            I = val_sum / count;
            err = fabs(I - analytic_I);
            is_finished = count > MAX_N_POINTS - step_n_points || err < eps;
        }
        MPI_Bcast(&is_finished, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (is_finished) {
            break;
        }
    }
    I *= volume;

    finish = MPI_Wtime();
    time = finish - start;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%f %f %d %f\n", I, err, count, max_time);
    }

    MPI_Finalize();
    return 0;
}