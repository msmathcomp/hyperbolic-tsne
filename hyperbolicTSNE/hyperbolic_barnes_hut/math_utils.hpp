#pragma once
#include <cmath>

#define M_UTILS_EPSILON 1e-5

namespace hyperbolic_utils {

    inline double sq_norm(const double& a, const double& b) {
        return a * a + b * b;
    }

    inline double poincare_to_klein(const double& a, const double& sq_n) {
        return 2 * a / (1 + sq_n);
    }

    inline double klein_to_poincare(const double& a, const double& sq_n) {
        return a / (1 + std::sqrt(1 - sq_n));
    }

    inline double lorentz_factor(const double& sq_n) {
        return 1 / std::sqrt(1 - sq_n);
    }
}