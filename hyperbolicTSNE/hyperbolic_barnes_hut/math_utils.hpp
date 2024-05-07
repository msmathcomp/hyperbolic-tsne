#pragma once
#include <cmath>
#include "point.hpp"

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

    static bool isBoxWithinUnitCircle(const double& min_bounds_x, const double& min_bounds_y, const double& max_bounds_x, const double& max_bounds_y) {
        double d1 = min_bounds_x*min_bounds_x + min_bounds_y*min_bounds_y;
        double d2 = max_bounds_x*max_bounds_x + min_bounds_y*min_bounds_y;
        double d3 = max_bounds_x*max_bounds_x + max_bounds_y*max_bounds_y;
        double d4 = min_bounds_x*min_bounds_x + max_bounds_y*max_bounds_y;
        return d1 < 1.0 && d2 < 1.0 && d3 < 1.0 && d4 < 1.0;
    }
}