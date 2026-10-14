// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak
//
// Trains a Hopfield on the 50 real exemplars for 25 passes and, for one
// exemplar, prints its energy before and after each pass and the sweeps its
// settle takes. Usage: hopfield_example [exemplars.bin]. Without the file,
// or when it cannot be opened, the exemplars are synthesized uniform in
// [-1, 1] from a fixed seed so the example runs from a clean clone.

#include "Hopfield.h"
#include "Rng.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// The operating point measured in examples/recall_benchmark.cpp; the
// reasoning is in docs/hopfield.md, "Operating points".
constexpr HopfieldConfig kConfig{
    .dim = 10,
    .gain = 0.03f,
    .sweep_budget = 64,
    .rest_tolerance = 0.01f,
    .eta = 0.003f,
    .input_gain = 20.f,
    .seed = 0,
    .distance = 5,
};

constexpr size_t kExemplars = 50;
constexpr size_t kIterations = 25;
constexpr size_t kReportExemplar = 2;
constexpr size_t kSites = 1u << kConfig.dim;
// HOPFIELD_DATA_DIR is the repository's data/ directory, set by CMake.
constexpr char kDefaultPath[] = HOPFIELD_DATA_DIR "/exemplars.bin";

void Enter(const char* stage)
{
    std::cout << '[' << stage << "] " << std::flush;
}

void Leave(const std::string& detail)
{
    std::cout << detail << std::endl;
}

// Returns the exemplars and, through @p source, where they came from.
std::vector<float> LoadExemplars(const char* path, std::string& source)
{
    const size_t count = kExemplars * kSites;
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::vector<float> synth(count);
        std::mt19937_64 rng(1234);
        for (auto& x : synth) x = static_cast<float>(RngUniform11(rng));
        source = std::string("could not open ") + path + "; synthesized from seed 1234";
        return synth;
    }
    source = path;

    const std::streamsize bytes = static_cast<std::streamsize>(count * sizeof(float));
    std::vector<float> exemplars(count);
    in.read(reinterpret_cast<char*>(exemplars.data()), bytes);
    if (in.gcount() != bytes)
        throw std::runtime_error("exemplar file is shorter than "
                                 + std::to_string(kExemplars) + " patterns of "
                                 + std::to_string(kSites) + " floats");

    in.peek();
    if (!in.eof())
        throw std::runtime_error("exemplar file is longer than "
                                 + std::to_string(kExemplars) + " patterns of "
                                 + std::to_string(kSites) + " floats");
    return exemplars;
}

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : kDefaultPath;
    Enter("initialization");
    Hopfield net(kConfig);
    Leave("dim " + std::to_string(net.Dim())
          + " N " + std::to_string(net.N())
          + " gain " + std::to_string(kConfig.gain)
          + " sweep budget " + std::to_string(kConfig.sweep_budget)
          + " rest tolerance " + std::to_string(kConfig.rest_tolerance)
          + " eta " + std::to_string(kConfig.eta)
          + " input gain " + std::to_string(kConfig.input_gain)
          + " distance " + std::to_string(kConfig.distance)
          + " degree " + std::to_string(net.Degree())
          + " seed " + std::to_string(kConfig.seed)
          + " iterations " + std::to_string(kIterations)
          + " report exemplar " + std::to_string(kReportExemplar));

    Enter("loading");
    std::string source;
    const std::vector<float> exemplars = LoadExemplars(path, source);
    Leave(std::to_string(kExemplars) + " exemplars, "
          + std::to_string(kSites) + " sites, " + source);

    if (kReportExemplar < 1 || kReportExemplar > kExemplars)
        throw std::runtime_error("report exemplar is outside the loaded set");

    for (size_t iteration = 0; iteration < kIterations; ++iteration) {
        for (size_t i = 0; i < kExemplars; ++i) {
            const std::span<const float> pattern(exemplars.data() + i * kSites, kSites);
            const bool report = i + 1 == kReportExemplar;
            double before = 0.0;
            if (report) {
                Enter("training");
                before = net.Energy(pattern);
            }
            net.Train(pattern);
            if (report) {
                const double after = net.Energy(pattern);
                net.Latch(pattern);
                Leave("iteration " + std::to_string(iteration + 1) + " of " + std::to_string(kIterations)
                      + " exemplar " + std::to_string(kReportExemplar)
                      + " energy " + std::to_string(before)
                      + " -> " + std::to_string(after)
                      + " sweeps " + std::to_string(net.SweepCount()));
            }
        }
    }

    Enter("done");
    Leave("trained");
    return 0;
}
