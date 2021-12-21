#pragma once

#include <string>

struct Config {
  public:
    double L[3];
    double T;
    int N[3];
    int periodic[3];
    int K;
    bool save_layers;
    int save_step;
    std::string layers_path;

    Config(const std::string &path);

    void print() const;
};
