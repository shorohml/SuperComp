#include "Solver.h"

int main(int argc, char **argv) {
    std::string config_path("config.ini");
    if (2 == argc) {
        config_path = std::string(argv[1]);
    }
    Solver solver(config_path, argc, argv);
    solver.run();
}