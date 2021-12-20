#include "Solver.h"

int main(int argc, char **argv) {
    Solver solver("config.ini", argc, argv);

    solver.run();
}