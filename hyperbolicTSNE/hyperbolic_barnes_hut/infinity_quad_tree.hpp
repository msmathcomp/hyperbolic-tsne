#pragma once
#include "cell.hpp"
#include "point.hpp"
#include "math_utils.hpp"
#include "centre_of_mass.hpp"
#include <stddef.h>
#include <vector>
#include <algorithm>
#include <iostream>

class InfinityQuadTree {
public: 
    InfinityQuadTree(std::vector<Point>& points) {
        rec_build_tree(points.begin(), points.end(), Point{-1, -1}, Point{1, 1}, 0);
    }
    InfinityQuadTree() {}

    std::vector<Cell> get_nodes() {
        return _nodes;
    }

    size_t approximate_centers_of_mass(const double& x, const double& y, const double& theta_sq, double* combined_results) const {
        return approximate_centers_of_mass(Point{x,y}, 0, theta_sq, combined_results, 0);
        // std::cout << combined_results.size() << std::endl;
    }

private:
    size_t rec_build_tree(const std::vector<Point>::iterator& begin_points, const std::vector<Point>::iterator& end_points, const Point& min_bounds, const Point& max_bounds, size_t depth){
        if (begin_points == end_points)
            return 0;

        Point bb_center{(max_bounds.x + min_bounds.x) / 2, (max_bounds.y + min_bounds.y) / 2};

        // If only 1 point left - it's a leaf
        if (begin_points + 1 == end_points) {
            // Unless the square is not within a unit circle completely
            if (isBoxWithinUnitCircle(min_bounds, max_bounds)) {
                size_t result_idx = _nodes.size();
                _nodes.emplace_back(Cell(depth, min_bounds, max_bounds));

                _nodes[result_idx].is_leaf = true;
                _nodes[result_idx].lorentz_factor = hyperbolic_utils::lorentz_factor((*begin_points).to_klein().sq_norm()); 
                _nodes[result_idx].barycenter = (*begin_points);
                _nodes[result_idx].cumulative_size = 1;
                return result_idx;
            }
        }
        // Split the points based on their location
        auto split_y = std::partition(begin_points, end_points, [bb_center](Point a){ return a.y < bb_center.y; });
        auto split_x_lower = std::partition(begin_points, split_y, [bb_center](Point a){ return a.x < bb_center.x; });
        auto split_x_upper = std::partition(split_y, end_points, [bb_center](Point a){ return a.x < bb_center.x; });

        // Recursively call on created partitions
        size_t child0_idx = rec_build_tree(split_y, split_x_upper, Point{min_bounds.x, bb_center.y}, Point{bb_center.x, max_bounds.y}, depth + 1);
        size_t child1_idx = rec_build_tree(split_x_upper, end_points, bb_center, max_bounds, depth + 1);
        size_t child2_idx = rec_build_tree(begin_points, split_x_lower, min_bounds, bb_center, depth + 1);
        size_t child3_idx = rec_build_tree(split_x_lower, split_y, Point{bb_center.x, min_bounds.y}, Point{max_bounds.x, bb_center.y}, depth + 1);
    
        size_t only_child = std::max(std::max(child0_idx, child1_idx), std::max(child2_idx, child3_idx));
        if (child0_idx + child1_idx + child2_idx + child3_idx == only_child) {
            return only_child;
        }

        size_t result_idx = _nodes.size();
        _nodes.emplace_back(Cell(depth, min_bounds, max_bounds));

        _nodes[result_idx].children_idx[0] = child0_idx;
        _nodes[result_idx].children_idx[1] = child1_idx;
        _nodes[result_idx].children_idx[2] = child2_idx;
        _nodes[result_idx].children_idx[3] = child3_idx;

        // If child_idx is 0, it means that there is no child in that sector of the cell
        double new_lorentz_factor = (child0_idx == 0 ? 0 : _nodes[child0_idx].lorentz_factor)
            + (child1_idx == 0 ? 0 : _nodes[child1_idx].lorentz_factor)
            + (child2_idx == 0 ? 0 : _nodes[child2_idx].lorentz_factor)
            + (child3_idx == 0 ? 0 : _nodes[child3_idx].lorentz_factor);

        Point new_barycenter_klein = ((child0_idx == 0 ? Point{0, 0} : _nodes[child0_idx].barycenter.to_klein() * _nodes[child0_idx].lorentz_factor)
            + (child1_idx == 0 ? Point{0, 0} : _nodes[child1_idx].barycenter.to_klein() * _nodes[child1_idx].lorentz_factor)
            + (child2_idx == 0 ? Point{0, 0} : _nodes[child2_idx].barycenter.to_klein() * _nodes[child2_idx].lorentz_factor)
            + (child3_idx == 0 ? Point{0, 0} : _nodes[child3_idx].barycenter.to_klein() * _nodes[child3_idx].lorentz_factor)) / new_lorentz_factor;

        _nodes[result_idx].barycenter = new_barycenter_klein.to_poincare();
        _nodes[result_idx].lorentz_factor = new_lorentz_factor;
        _nodes[result_idx].cumulative_size = end_points - begin_points;

        return result_idx;
    }

    static bool isBoxWithinUnitCircle(const Point& min_bounds, const Point& max_bounds) {
        return hyperbolic_utils::isBoxWithinUnitCircle(min_bounds.x, min_bounds.y, max_bounds.x, max_bounds.y);
    }

    size_t approximate_centers_of_mass(const Point& target, size_t cell_idx, double theta_sq, double* combined_results, size_t idx) const {
        auto& current_cell = _nodes[cell_idx];

        if (current_cell.is_leaf && std::fabs(target.x - current_cell.barycenter.x) < 1e-5 && std::fabs(target.y - current_cell.barycenter.y) < 1e-5) 
            return idx;

        double distance_to_target = target.distance_to_point_poincare(_nodes[cell_idx].barycenter);
        double distance_squared = distance_to_target * distance_to_target;
        combined_results[cell_idx * 4 + 2] = distance_squared;

        // Check the stop condition
        if (_nodes[cell_idx].is_leaf || (!current_cell.contains_infinity && (current_cell.max_distance_within_squared / distance_squared < theta_sq))) {
            combined_results[cell_idx*4] = _nodes[cell_idx].barycenter.x;
            combined_results[cell_idx*4 + 1] = _nodes[cell_idx].barycenter.y;
            combined_results[cell_idx*4 + 2] = distance_squared;
            combined_results[cell_idx*4 + 3] = _nodes[cell_idx].cumulative_size;
            return idx + 4;
        }
        
        combined_results[cell_idx*4 + 3] = 0;
        // If stop condition wasn't triggered - go deeper and combine results
        for(int i = 0; i < 4; ++i) {
            if (_nodes[cell_idx].children_idx[i] != 0) {
                idx = approximate_centers_of_mass(target, _nodes[cell_idx].children_idx[i], theta_sq, combined_results, idx);
            }
        }
        return idx;
    }

    std::vector<Cell> _nodes;
};