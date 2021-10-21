#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const double ANALITIC_I = 0.0;
const double COORDS_MIN[3] = {0.0, 0.0, 0.0};
const double COORDS_MAX[3] = {1.0, 1.0, 1.0};
const int MAX_STEPS = 10000;

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
    double coords[3];
    double eps, time;
    double val, val_sum = 0.0;
    double I, err;
    double start, finish;
    double max_time;

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

    while (1) {
        // generate random coordinates
        for (int j = 0; j < 3; ++j) {
            coords[j] = (double)rand() / INT_MAX;
            coords[j] = coords[j] * (COORDS_MAX[j] - COORDS_MIN[j]) + COORDS_MIN[j];
        }

        // compute and gather function values
        val = f(coords[0], coords[1], coords[2]);
        MPI_Reduce(&val, &val_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // check finishing criterion
        if (rank == 0) {
            count += size;
            I = val_sum * 8 / count;
            err = fabs(I - ANALITIC_I);
            is_finished = count > MAX_STEPS - size || err < eps;
        }
        MPI_Bcast(&is_finished, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (is_finished) {
            break;
        }
    }

    finish = MPI_Wtime();
    time = finish - start;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%f %f %d %f\n", I, err, count, max_time);
    }

    MPI_Finalize();
    return 0;
}