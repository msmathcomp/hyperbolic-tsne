#pragma once
#include "point.hpp"
#include <stddef.h>
#include <vector>

struct Cell{
    size_t parent_idx;
    std::vector<size_t> children_idx;

    bool is_leaf;
    size_t cumulative_size;
    size_t depth;

    Point center;
    Point min_bounds;
    Point max_bounds;

    double max_distance_within_squared;

    Point barycenter;
    double lorentz_factor;

    Cell(size_t depth_, const Point& min_bounds_, const Point& max_bounds_) : 
        parent_idx(0), 
        children_idx{{0, 0, 0, 0}}, 
        is_leaf(false), 
        cumulative_size(0), 
        depth(depth_),
        center{(min_bounds_ + max_bounds_) / 2},
        min_bounds{min_bounds_}, 
        max_bounds{max_bounds_},
        barycenter{(min_bounds_ + max_bounds_) / 2},
        lorentz_factor(1) 
        /*
        a               max_bounds
        min_bounds      b
        
        */

        {
            Point a = Point{min_bounds.x, max_bounds.y};
            Point b = Point{min_bounds.y, max_bounds.x};
            max_distance_within_squared = fmax(
                fmax(a.distance_to_point_poincare(b), min_bounds.distance_to_point_poincare(max_bounds)),
                fmax(fmax(a.distance_to_point_poincare(min_bounds), min_bounds.distance_to_point_poincare(b)),
                fmax(a.distance_to_point_poincare(max_bounds), max_bounds.distance_to_point_poincare(b)))
                );
            max_distance_within_squared *= max_distance_within_squared;
        }
};