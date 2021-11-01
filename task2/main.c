#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const double ANALITIC_I = 0.06225419868854213;
const double COORDS_MIN[3] = {0.0, 0.0, 0.0};
const double COORDS_MAX[3] = {1.0, 1.0, 1.0};
const int N_STEP_POINTS = 640;
const int MAX_N_STEPS = 1000000;

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
    int count, is_finished;
    int n_runs, n_proc_step_points, n_step_points;
    double coords[3], coords_diff[3];
    double val, val_sum;
    double volume, analytic_I;
    double I, err, eps;
    double start, finish;
    double curr_time, max_time;

    if (argc != 3) {
        printf("Usage: ./main eps n_runs\n");
        return 0;
    }
    // target approximation error
    eps = atof(argv[1]);
    if (eps <= 0.0) {
        printf("Invalid eps\n");
        return 0;
    }
    // number of runs with different seeds
    n_runs = atoi(argv[2]);
    if (n_runs <= 0) {
        printf("Invalid n_runs\n");
        return 0;
    }

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("%d %f\n", size, eps);

        curr_time = (double)time(NULL);
        curr_time = fmod(curr_time, (double)INT_MAX - n_runs * size);
    }
    MPI_Bcast(&curr_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int run_idx = 0; run_idx < n_runs; ++run_idx) {
        start = MPI_Wtime();

        // set random seed
        srand((int)curr_time + run_idx * size + rank);

        // compute domain volume
        if (rank == 0) {
            volume = 1.0;
            for (int i = 0; i < 3; ++i) {
                volume *= COORDS_MAX[i] - COORDS_MIN[i];
            }
            // for a small speedup
            analytic_I = ANALITIC_I / volume;
        }
        n_proc_step_points = N_STEP_POINTS / size;
        n_step_points = n_proc_step_points * size;

        // precompute COORDS_MAX - COORDS_MAX for a small speedup
        for (int i = 0; i < 3; ++i) {
            coords_diff[i] = COORDS_MAX[i] - COORDS_MIN[i];
        }
        is_finished = 0;
        count = 0;
        val_sum = 0.0;
        while (1) {
            val = 0.0;
            for (int i = 0; i < n_proc_step_points; ++i) {
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
                ++count;
                I = val_sum / (count * n_step_points);
                err = fabs(I - analytic_I);
                is_finished = err < eps || count >= MAX_N_STEPS;
            }
            MPI_Bcast(&is_finished, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (is_finished) {
                break;
            }
        }
        I *= volume;

        finish = MPI_Wtime();
        curr_time = finish - start;
        MPI_Reduce(&curr_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("%f %f %d %f\n", I, err, count * n_step_points, max_time);
        }
    }

    MPI_Finalize();
    return 0;
}