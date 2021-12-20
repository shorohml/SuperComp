#include "Config.h"
#include "INIReader.h"
#include <iostream>
#include <stdexcept>

Config::Config(const std::string &path) {
    INIReader reader(path);

    if (0 != reader.ParseError()) {
        throw std::runtime_error("Can't read config");
    }

    L[0] = reader.GetReal("Solver", "L_x", 1.0);
    L[1] = reader.GetReal("Solver", "L_y", 1.0);
    L[2] = reader.GetReal("Solver", "L_z", 1.0);
    T = reader.GetReal("Solver", "T", 0.025);
    N[0] = reader.GetInteger("Solver", "N_x", 128);
    N[1] = reader.GetInteger("Solver", "N_y", 128);
    N[2] = reader.GetInteger("Solver", "N_z", 128);
    K = reader.GetInteger("Solver", "K", 20);

    save_layers = reader.GetBoolean("Solver", "save_layers", false);
    save_step = reader.GetInteger("Solver", "save_step", 5);
    layers_path = reader.Get("Solver", "layers_path", "./layers");
    if (layers_path.size() && layers_path[layers_path.size() - 1] != '/') {
        layers_path += "/";
    }
}

void Config::print() const {
    std::cout << "Config:" << std::endl;
    std::cout << "L_x: " << L[0] << std::endl;
    std::cout << "L_y: " << L[1] << std::endl;
    std::cout << "L_z: " << L[2] << std::endl;
    std::cout << "T: " << T << std::endl;
    std::cout << "N_x: " << N[0] << std::endl;
    std::cout << "N_y: " << N[1] << std::endl;
    std::cout << "N_z: " << N[2] << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "save_layers: " << save_layers << std::endl;
    std::cout << "save_step: " << save_step << std::endl;
    std::cout << "layers_path: " << layers_path << std::endl << std::endl;
}