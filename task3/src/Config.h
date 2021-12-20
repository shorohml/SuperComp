#pragma once

#include "INIReader.h"
#include <string>
#include <iostream>

struct Config {
  public:
    double L_x;
    double L_y;
    double L_z;
    double T;
    int N;
    int K;
    bool save_layers;
    int save_step;
    std::string layers_path;

    Config(const std::string &path);

    void print() const;
};

Config::Config(const std::string &path) {
    INIReader reader(path);

    if (0 != reader.ParseError()) {
        throw std::runtime_error("Can't read config");
    }

    L_x = reader.GetReal("Solver", "L_x", 1.0);
    L_y = reader.GetReal("Solver", "L_y", 1.0);
    L_z = reader.GetReal("Solver", "L_z", 1.0);
    T = reader.GetReal("Solver", "T", 0.025);
    N = reader.GetInteger("Solver", "N", 128);
    K = reader.GetInteger("Solver", "K", 20);

    save_layers = reader.GetBoolean("Solver", "save_layers", false);
    save_step = reader.GetInteger("Solver", "save_step", 5);
    layers_path = reader.Get("Solver", "layers_path", "./layers");
    if (layers_path.size() && layers_path[layers_path.size() - 1] != '/') {
        layers_path += "/";
    }
}

void Config::print() const {
    std::cout << "Config:" << std::endl << std::endl;
    std::cout << "L_x: " << L_x << std::endl;
    std::cout << "L_y: " << L_y << std::endl;
    std::cout << "L_z: " << L_z << std::endl;
    std::cout << "T: " << T << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "save_layers: " << save_layers << std::endl;
    std::cout << "save_step: " << save_step << std::endl << std::endl;
    std::cout << "layers_path: " << layers_path << std::endl << std::endl;
}
