#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const double L_x = 1.0;
const double L_y = 1.0;
const double L_z = 1.0;
const double T = 1.0;
const size_t N = 128;
const size_t K = 400;
const bool SAVE_LAYERS = true;
const size_t SAVE_STEPS = 100;

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

int main(int argc, char **argv) {
    double a_t = M_PI * sqrt(4 / (L_x * L_x) + 16 / (L_y * L_y) + 36 / (L_z * L_z));
    double delta, max_err;
    size_t idx;
    const size_t grid_size = (N + 1) * (N + 1) * (N + 1);
    const size_t nn = (N + 1) * (N + 1);
    const double tau = T / K;
    const double h_x = L_x / N;
    const double h_y = L_x / N;
    const double h_z = L_x / N;
    std::vector<std::vector<double>> layers(3, std::vector<double>(grid_size));
    std::vector<double> tmp_layer, errs(grid_size);

    // compute u_0
    for (size_t i = 0; i < N + 1; ++i) {
        for (size_t j = 0; j < N + 1; ++j) {
            for (size_t k = 0; k < N + 1; ++k) {
                layers[0][i * (N + 1) * (N + 1) + j * (N + 1) + k] =
                    u(L_x * i / N, L_y * j / N, L_z * k / N, a_t, 0.0);
            }
        }
    }

    // compute u_1 inner values
    for (size_t i = 1; i < N; ++i) {
        for (size_t j = 1; j < N; ++j) {
            for (size_t k = 1; k < N; ++k) {
                idx = i * nn + j * (N + 1) + k;
                delta = 0.0;
                delta +=
                    (layers[0][idx - nn] - 2 * layers[0][idx] + layers[0][idx + nn]) / (h_x * h_x);
                delta += (layers[0][idx - N - 1] - 2 * layers[0][idx] + layers[0][idx + N + 1]) /
                         (h_y * h_y);
                delta +=
                    (layers[0][idx - 1] - 2 * layers[0][idx] + layers[0][idx + 1]) / (h_z * h_z);
                layers[1][idx] = layers[0][idx] + (tau * tau) / 2.0 * delta;
            }
        }
    }
    // compute u_1 outer values
    for (size_t i = 0; i < N + 1; ++i) {
        for (size_t j = 0; j < N + 1; ++j) {
            layers[1][i * nn + j * (N + 1)] = u(L_x * i / N, L_y * j / N, 0.0, a_t, tau);
            layers[1][i * nn + j * (N + 1) + N] = u(L_x * i / N, L_y * j / N, L_z, a_t, tau);
        }
    }
    for (size_t i = 0; i < N + 1; ++i) {
        for (size_t k = 0; k < N + 1; ++k) {
            layers[1][i * nn + k] = u(L_x * i / N, 0.0, L_z * k / N, a_t, tau);
            layers[1][i * nn + N * (N + 1) + k] = u(L_x * i / N, L_y, L_z * k / N, a_t, tau);
        }
    }
    for (size_t j = 0; j < N + 1; ++j) {
        for (size_t k = 0; k < N + 1; ++k) {
            layers[1][j * (N + 1) + k] = u(0.0, L_y * j / N, L_z * k / N, a_t, tau);
            layers[1][N * nn + j * (N + 1) + k] = u(L_x, L_y * j / N, L_z * k / N, a_t, tau);
        }
    }

    // compute approximation
    for (size_t t_step = 2; t_step < K + 1; ++t_step) {
        // inner values
        for (size_t i = 1; i < N; ++i) {
            for (size_t j = 1; j < N; ++j) {
                for (size_t k = 1; k < N; ++k) {
                    idx = i * nn + j * (N + 1) + k;
                    delta = 0.0;
                    delta += (layers[1][idx - nn] - 2 * layers[1][idx] + layers[1][idx + nn]) /
                             (h_x * h_x);
                    delta +=
                        (layers[1][idx - N - 1] - 2 * layers[1][idx] + layers[1][idx + N + 1]) /
                        (h_y * h_y);
                    delta += (layers[1][idx - 1] - 2 * layers[1][idx] + layers[1][idx + 1]) /
                             (h_z * h_z);
                    layers[2][idx] = 2 * layers[1][idx] - layers[0][idx] + tau * tau * delta;
                }
            }
        }
        // outer values
        for (size_t i = 0; i < N + 1; ++i) {
            for (size_t j = 0; j < N + 1; ++j) {
                layers[2][i * nn + j * (N + 1)] =
                    u(L_x * i / N, L_y * j / N, 0.0, a_t, tau * t_step);
                layers[2][i * nn + j * (N + 1) + N] =
                    u(L_x * i / N, L_y * j / N, L_z, a_t, tau * t_step);
            }
        }
        for (size_t i = 0; i < N + 1; ++i) {
            for (size_t k = 0; k < N + 1; ++k) {
                layers[2][i * nn + k] = u(L_x * i / N, 0.0, L_z * k / N, a_t, tau * t_step);
                layers[2][i * nn + N * (N + 1) + k] =
                    u(L_x * i / N, L_y, L_z * k / N, a_t, tau * t_step);
            }
        }
        for (size_t j = 0; j < N + 1; ++j) {
            for (size_t k = 0; k < N + 1; ++k) {
                layers[2][j * (N + 1) + k] = u(0.0, L_y * j / N, L_z * k / N, a_t, tau * t_step);
                layers[2][N * nn + j * N + k] = u(L_x, L_y * j / N, L_z * k / N, a_t, tau * t_step);
            }
        }
        // compute max absolute error
        max_err = 0.0;
        for (size_t i = 0; i < N + 1; ++i) {
            for (size_t j = 0; j < N + 1; ++j) {
                for (size_t k = 0; k < N + 1; ++k) {
                    idx = i * nn + j * (N + 1) + k;
                    errs[idx] = fabs(layers[2][idx] -
                                     u(L_x * i / N, L_y * j / N, L_z * k / N, a_t, tau * t_step));
                    if (errs[idx] > max_err) {
                        max_err = errs[idx];
                    }
                }
            }
        }
        if (SAVE_LAYERS) {
            if (t_step % SAVE_STEPS == 0) {
                save_layer(layers[2], "layer" + std::to_string(t_step) + ".bin");
                save_layer(errs, "errs" + std::to_string(t_step) + ".bin");
            }
        }
        tmp_layer = std::move(layers[0]);
        layers[0] = std::move(layers[1]);
        layers[1] = std::move(layers[2]);
        layers[2] = std::move(tmp_layer);
        std::cout << "layer " << t_step << " error: " << max_err << std::endl;
    }
    return 0;
}