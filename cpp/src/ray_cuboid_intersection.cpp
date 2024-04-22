#include "ray_cuboid_intersection.h"

void coord_transform(float *vec3,
                     float angle_rad, float x_shift, float y_shift, float trans_x, float trans_y)
{
    // translational, image space origin not centered
    vec3[0] = vec3[0] - trans_x;
    vec3[1] = vec3[1] - trans_y;
    // Rotational
    // Angle in radians
    vec3[0] = vec3[0] * cos(angle_rad) + vec3[1] * sin(angle_rad);
    vec3[1] = vec3[1] * cos(angle_rad) - vec3[0] * sin(angle_rad);
    // Translational transformation
    vec3[0] = vec3[0] - x_shift;
    // vec3[0] = vec3[0]
    vec3[1] = vec3[1] + y_shift;
}

float center_to_line_dist(float *cuboid,
                          float *pointA,
                          float *pointB)
{

    // Vector OA, OB
    // O is the center of the cuboid
    float oa[3], ob[3];
    float ab_sqr = 0;
    for (int i = 0; i < 3; i++)
    {
        float component_i = 0.5 * (cuboid[2 * i] + cuboid[2 * i + 1]);
        oa[i] = component_i - pointA[i];
        ob[i] = component_i - pointB[i];
        ab_sqr += (pointB[i] - pointA[i]) * (pointB[i] - pointA[i]);
    }

    // OA X OB coss product x, y, z components
    float cpx = oa[1] * ob[2] - oa[2] * ob[1];
    float cpy = oa[2] * ob[0] - oa[0] * ob[2];
    float cpz = oa[0] * ob[1] - oa[1] * ob[0];
    float result = sqrt((cpx * cpx + cpy * cpy + cpz * cpz) / ab_sqr);

    return result;
}
float get_intersection(float *cuboid, float *pA, float *pB)
{
    const float epsilon = 1e-8;
    float t_list[2] = {0, 0};
    unsigned int counter = 0;
    float distance = center_to_line_dist(cuboid, pA, pB);
    float diagnal_sqr = 0;

    for (int i = 0; i < 3; i++)
    {
        diagnal_sqr += (cuboid[2 * i + 1] - cuboid[2 * i]) *
                       (cuboid[2 * i + 1] - cuboid[2 * i]);
    }
    // printf("O-AB distance: %.3f\n",distance);
    // printf("Cuboid half-diagnal: %.3f\n",sqrt(diagnal_sqr) * 0.5);

    if (distance > sqrt(diagnal_sqr) * 0.5)
    {
        return 0;
    }
    // Case 1: intersects on face x = x_0 or face x = x_1
    // Note that A_x never equals B_x.
    for (int i = 0; i < 2; i++)
    {
        float yx =
            (cuboid[i] - pA[0]) / (pB[0] - pA[0]) * (pB[1] - pA[1]) + pA[1];
        float zx =
            (cuboid[i] - pA[0]) / (pB[0] - pA[0]) * (pB[2] - pA[2]) + pA[2];
        if ((yx - cuboid[2]) * (yx - cuboid[3]) < epsilon and
            (zx - cuboid[4]) * (zx - cuboid[5]) < epsilon)
        {
            float t = (cuboid[i] - pA[0]) / (pB[0] - pA[0]);
            if (t < 1)
            {
                t_list[counter] = t;
                counter++;
            }
        }
    }
    // Case 2 : intersects on face y = y_0 or face y = y_1
    // Note : we exclude the case when A_y equals B_y
    if (pB[1] - pA[1] != 0)
    {
        for (int i = 2; i < 4; i++)
        {
            float xy =
                (cuboid[i] - pA[1]) / (pB[1] - pA[1]) * (pB[0] - pA[0]) + pA[0];
            float zy =
                (cuboid[i] - pA[1]) / (pB[1] - pA[1]) * (pB[2] - pA[2]) + pA[2];
            if ((xy - cuboid[0]) * (xy - cuboid[1]) < epsilon and
                (zy - cuboid[4]) * (zy - cuboid[5]) < epsilon)
            {
                float t = (cuboid[i] - pA[1]) / (pB[1] - pA[1]);
                if (t < 1)
                {
                    t_list[counter] = t;
                    counter++;
                }
            }
        }
    }

    // Case 3 : intersects on face z = z_0 or face z = z_1
    // Note : we exclude the case when A_z equals B_z
    if (pB[1] - pA[1] != 0)
    {
        for (int i = 4; i < 6; i++)
        {
            float xz =
                (cuboid[i] - pA[2]) / (pB[2] - pA[2]) * (pB[0] - pA[0]) + pA[0];
            float yz =
                (cuboid[i] - pA[2]) / (pB[2] - pA[2]) * (pB[1] - pA[1]) + pA[1];
            if ((xz - cuboid[0]) * (xz - cuboid[1]) < epsilon and
                (yz - cuboid[2]) * (yz - cuboid[3]) < epsilon)
            {
                float t = (cuboid[i] - pA[2]) / (pB[2] - pA[2]);
                if (t < 1)
                {
                    t_list[counter] = t;
                    counter++;
                }
            }
        }
    }
    float result = 0.0;
    if (counter > 1)
    {
        float coords[2][3] = {{pA[0], pA[1], pA[2]}, {pA[0], pA[1], pA[2]}};
        for (int i = 0; i < counter; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                coords[i][j] = t_list[i] * (pB[j] - pA[j]) + pA[j];
            }
        }
        for (int i = 0; i < 3; i++)
        {
            result += (coords[0][i] - coords[1][i]) * (coords[0][i] - coords[1][i]);
        }
    }
    result = sqrt(result);
    return result;
}