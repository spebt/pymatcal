// #include <vector>
// #include <limits>
#include <cmath>
// #include <iostream>
#include <stdio.h>

float center_to_line_dist(float *,
                          float *,
                          float *);
float get_intersection(float *, float *, float *) asm ("get_intersection");
float get_intersection(float *, float *, float *);
