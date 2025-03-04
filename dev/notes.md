# Timing of the algorithm to reduce the number of geometries for raytracing

## Introduction

The algorithm is base on convex hull and the idea is to reduce the number of geometries to be raytraced. All geometries are represented by a set of rectangles with 4 points each. The algorithm is implemented in Python and Pytorch.

## Timing results

### Computing environment

CPU Information:

- Number of threads:       8  
- Model name:             Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
  - Thread(s) per core:   2
  - Core(s) per socket:   4
  - Socket(s):            1

Memory Information:

- Total memory: 64 GB
- Details:
  - 4 x 16 GB
  - Type: DDR4
  - Speed: 2133 MHz
  - Brand: G-Skill

### Comparison

- Iteration exection time mean: 0.0745 seconds
- 864 iterations, using ``for`` loop

Unit: seconds

| Mean    | STD    | MAX    | MIN    | SUM     |
|---------|--------|--------|--------|---------|
| 0.0745  | 0.0025 | 0.0971 | 0.0713 | 64.3311 |

- 864 iterations, NOT using ``for`` loop

Unit: seconds

| Mean    | STD    | MAX    | MIN    | SUM     |
|---------|--------|--------|--------|---------|
| 0.0011  | 0.0002 | 0.0025 | 0.0010 | 0.9693  |

## Conclusion

The algorithm is faster when not using the ``for`` loop. The execution time is reduced from 64.3311 seconds to 0.9693 seconds, which is 66 times faster.
