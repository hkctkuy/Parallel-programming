#include <cmath>
#include <iostream>
#include <string>

#ifdef DEBUG
#include <cassert>
#endif

#include <mpi.h>

#include "factory.hpp"

#define _USE_MATH_DEFINES

int main(int argc, char** argv) {
    // Help
    if (argc < 5 || argc > 6) {
        std::cout << "Usage: " << argv[0] << " N px py pz [cpu|gpu]\n"
            << "N is a Node Number per Side\n"
            << "px is a x axis decomposition param\n"
            << "py is a y axis decomposition param\n"
            << "pz is a z axis decomposition param\n"
            << "cpu|gpu are possible calculation modes: CPU-only (default) or GPU-only\n";
        return 1;
    }
    // Get args
    auto N = std::stoll(argv[1]);
    auto px = std::stoi(argv[2]);
    auto py = std::stoi(argv[3]);
    auto pz = std::stoi(argv[4]);
    bool gpu = false;
    if (argc == 6) {
        auto mode = std::string(argv[5]);
        if (mode == "gpu") {
            gpu = true;
        } else if (mode != "cpu") {
            std::cerr << "Bad calculation mode: " << mode << "\n"
                      << "Try \"cpu\" or \"gpu\"\n";
            return 1;
        }
    }
    size_t K = 20;
    // Init MPI
    // NOTE: I would like to move it inside solver maker using std::optional
    // But we have to deal with c++11 in patch version
    // So keep it here is the most logical decision
    int status, pn, pr;
    status = MPI_Init(&argc, &argv);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to init MPI: " << status << "\n";
        return 1;
    }
    status = MPI_Comm_size(MPI_COMM_WORLD, &pn);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to get process number: " << status << "\n";
        return 1;
    }
    status = MPI_Comm_rank(MPI_COMM_WORLD, &pr);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to get process rank: " << status << "\n";
        return 1;
    }
    // Solve
    for(auto L: {1.0, M_PI}) {
        if (pr == 0) {
            std::cout << "L = " << L << " "
                      << "N = " << N << " "
                      << "K = " << K << "\n";
        }
        auto solver = solver::SolverFactory(L, N, K, px, py, pz, pn, pr, gpu).make_solver();
        auto [_, err, time] = solver->solve();
        if (pr == 0) {
            std::cout << "Err:                " << err           << "\n"
                      << "Total Time:         " << time.total    << "\n"
                      << "Calc inner Time:    " << time.inner    << "\n"
                      << "Calc boundary Time: " << time.boundary << "\n"
                      << "Pack Time:          " << time.pack     << "\n"
                      << "Wait Time:          " << time.wait     << "\n"
                      << "Load Time:          " << time.load     << "\n"
                      << "Store Time:         " << time.store    << "\n"
                      << "\n";
        }
    }
    MPI_Finalize();
    return 0;
}
