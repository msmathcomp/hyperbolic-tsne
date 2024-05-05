#pragma once
#include "math_utils.hpp"

struct Point {
    double x;
    double y;

    inline double sq_norm() const {
        return x * x + y * y;
    }

    Point operator*(double a) const {
        return Point{x * a, y * a};
    }

    Point operator/(double a) const {
        return Point{x / a, y / a};
    }

    Point operator+(const Point& b) const {
        return Point{x + b.x, y + b.y};
    }

    Point to_klein() const {
        double norm = sq_norm();
        return Point{hyperbolic_utils::poincare_to_klein(x, norm), hyperbolic_utils::poincare_to_klein(y, norm)};
    }

    Point to_poincare() const {
        double norm = sq_norm();
        return Point{hyperbolic_utils::klein_to_poincare(x, norm), hyperbolic_utils::klein_to_poincare(y, norm)};
    }

    double distance_to_point_poincare(Point b) const {
        if (std::fabs(x - b.x) <= M_UTILS_EPSILON && std::fabs(y - b.y) <= M_UTILS_EPSILON)
            return 0;

        double uv2 = ((x - b.x) * (x - b.x)) + ((y - b.y) * (y - b.y));
        double u_sq = sq_norm();
        double v_sq = b.sq_norm();
        double result = std::acosh( 1. + 2. * uv2 / ( (1. - u_sq) * (1. - v_sq)));

        return result;
    }
};

        