#pragma once
#include "point.hpp"

struct CenterOfMass {
    Point position;
    size_t number_of_accumulated_points;
    double distance_to_target;

    
};