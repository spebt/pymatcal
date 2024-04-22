#include <ray_cuboid_intersection.h>
#include <iostream>

#include <vector>
#include <yaml-cpp/yaml.h>
int main(int argn, char **argvs)
{
    std::string config_fname = argvs[1];
    std::cout << "Using config file: " << config_fname << "\n";
    YAML::Node config = YAML::LoadFile(config_fname);
    //   const std::string username = config["username"].as<std::string>();
    const std::vector<std::vector<float>> geomDefinition =
        config["detector geometry"].as<std::vector<std::vector<float>>>();
    const std::vector<float> detNsubs = config["detector"]["crystal n subdivision xyz"].as<std::vector<float>>();
    const std::vector<unsigned int> imgNsubs = config["image"]["subdivision xyz"].as<std::vector<unsigned int>>();
    const std::vector<float> imageDimensions = config["image"]["dimension xyz"].as<std::vector<float>>();
    const std::vector<float> imageNVoxelsPerMM = config["image"]["voxel per mm xyz"].as<std::vector<float>>();

    // Calculate Image space N subdivision and size.
    const float angle_rad = config["image"]["detector rotation"].as<float>();
    const float x_shift = config["image"]["detector x-shift"].as<float>();
    unsigned int imageNVoxels = 1;
    for (int idx : {0, 1, 2})
    {
        imageNVoxels *= imageDimensions[idx] * imageNVoxelsPerMM[idx];
    }
    float yMin=1000,yMax=0;
    for(auto cuboid : geomDefinition){
        float yLow = cuboid[2];
        float yUp = cuboid[3];
        if (yLow<yMin) yMin = yLow;
        if (yUp>yMax) yMax = yUp;

    }
    float trans_x = imageDims[0] * 0.5
    float trans_y = imageDims[1] * 0.5
    y_shift = 0.5 * (yMax - yMin)
    imageNxyz = imageDims * imageVxpms
    mmPerVoxel = 1.0 / imageVxpms
    // std::cout << imageNVoxels << "\n";
    // std::cout << get_intersection(cuboid.data(), pA.data(), pB.data()) << std::endl;
    return 0;
}