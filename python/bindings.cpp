// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak
//
// Thin pybind11 surface for HypercubeHopfield1984. Ergonomics (argument
// coercion, pickle, docs) live in hypercube_hopfield1984/__init__.py.
//
// Buffer rules, from Hopfield.h: the class never keeps a pointer into a
// caller's buffer, so a forcecast temporary is a safe exposure; the spans
// Latch and Cube return and the reference Weights returns point into the
// object and are overwritten in place, so every output is copied to a fresh
// numpy array. Latch, Train, Energy, and LoadWeights touch only the object's
// own storage, so they run with the GIL released.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "../Hopfield.h"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

std::span<const float> AsSpan(const py::buffer_info& buf)
{
    return {static_cast<const float*>(buf.ptr), static_cast<size_t>(buf.size)};
}

py::array_t<float> ToArray(std::span<const float> v)
{
    py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
    if (!v.empty())
        std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
    return arr;
}

} // namespace

PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeHopfield1984: Hopfield's graded associative memory on "
              "a Boolean hypercube";
#ifndef HYPERCUBE_HOPFIELD1984_VERSION
#  error "HYPERCUBE_HOPFIELD1984_VERSION must be set by CMake from hypercube_hopfield1984/_version.py"
#endif
    m.attr("__version__") = HYPERCUBE_HOPFIELD1984_VERSION;

    // One de-templated class: dim is a runtime constructor argument.
    py::class_<Hopfield>(m, "_Core")
        .def(py::init([](size_t dim, float gain, size_t sweep_budget, float eta,
                         float rest_tolerance, float input_gain, uint64_t seed,
                         float init_scale, size_t distance) {
            HopfieldConfig cfg;
            cfg.dim            = dim;
            cfg.gain           = gain;
            cfg.sweep_budget   = sweep_budget;
            cfg.rest_tolerance = rest_tolerance;
            cfg.eta            = eta;
            cfg.input_gain     = input_gain;
            cfg.seed           = seed;
            cfg.init_scale     = init_scale;
            cfg.distance       = distance;
            return Hopfield(cfg);
        }),
            py::arg("dim"),
            py::arg("gain"),
            py::arg("sweep_budget"),
            py::arg("eta"),
            py::arg("rest_tolerance") = 0.f,
            py::arg("input_gain")     = 1.f,
            py::arg("seed")           = 0ULL,
            py::arg("init_scale")     = 0.f,
            py::arg("distance")       = 1ULL)

        .def("latch", [](Hopfield& self, FloatArray exposure) {
            const auto buf = exposure.request();
            std::span<const float> cube;
            {
                py::gil_scoped_release release;
                cube = self.Latch(AsSpan(buf));
            }
            return ToArray(cube);
        }, py::arg("exposure"),
           "Load the exposure as starting state and input current, settle, "
           "and return the settled cube (length N, a copy).")

        .def("train", [](Hopfield& self, FloatArray exposure) {
            const auto buf = exposure.request();
            py::gil_scoped_release release;
            self.Train(AsSpan(buf));
        }, py::arg("exposure"),
           "Add eta * e_v * e_u to every edge and clamp to [-1, 1].")

        .def("energy", [](const Hopfield& self, FloatArray state, FloatArray input) {
            const auto sbuf = state.request();
            const auto ibuf = input.request();
            py::gil_scoped_release release;
            return self.Energy(AsSpan(sbuf), AsSpan(ibuf));
        }, py::arg("state"), py::arg("input"),
           "Hopfield's energy of state under the current weights with input "
           "applied as the current input_gain * input_v.")

        .def("cube", [](const Hopfield& self) {
            return ToArray(self.Cube());
        }, "The settled cube from the most recent latch (a copy).")

        .def("weights", [](const Hopfield& self) {
            return ToArray(self.Weights());
        }, "All weights, row-major N x degree (a copy).")

        .def("load_weights", [](Hopfield& self, FloatArray weights) {
            const auto buf = weights.request();
            py::gil_scoped_release release;
            self.LoadWeights(AsSpan(buf));
        }, py::arg("weights"),
           "Replace every weight; length must equal N * degree and the two "
           "rows of each edge must agree.")

        .def_property_readonly("latched", &Hopfield::Latched)
        .def_property_readonly("sweep_count", &Hopfield::SweepCount)
        .def_property_readonly("at_rest", &Hopfield::AtRest)
        .def_property_readonly("dim", &Hopfield::Dim)
        .def_property_readonly("N", &Hopfield::N)
        .def_property_readonly("distance", &Hopfield::Distance)
        .def_property_readonly("degree", &Hopfield::Degree)
        .def_property_readonly("num_weights", [](const Hopfield& self) {
            return self.Weights().size();
        })
        .def_property_readonly("gain", [](const Hopfield& self) { return self.Config().gain; })
        .def_property_readonly("sweep_budget", [](const Hopfield& self) { return self.Config().sweep_budget; })
        .def_property_readonly("rest_tolerance", [](const Hopfield& self) { return self.Config().rest_tolerance; })
        .def_property_readonly("eta", [](const Hopfield& self) { return self.Config().eta; })
        .def_property_readonly("input_gain", [](const Hopfield& self) { return self.Config().input_gain; })
        .def_property_readonly("seed", [](const Hopfield& self) { return self.Config().seed; })
        .def_property_readonly("init_scale", [](const Hopfield& self) { return self.Config().init_scale; });
}
