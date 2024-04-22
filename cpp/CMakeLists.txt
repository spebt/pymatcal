cmake_minimum_required(VERSION 3.0.0)
# project(
#   ray_cuboid_intersection
#   VERSION 0.1.0
#   LANGUAGES C CXX)

project(
  ray_cuboid_intersection
  VERSION 0.1.0
)
find_package(yaml-cpp REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# include(CTest)
# enable_testing()
# include(CheckLanguage)

include_directories(src)
add_library(ray_cuboid_intersection SHARED src/ray_cuboid_intersection.cpp)
set_target_properties(ray_cuboid_intersection PROPERTIES LINKER_LANGUAGE C)
add_executable(main src/main.cpp)
target_link_libraries(main ray_cuboid_intersection)
target_link_libraries(main yaml-cpp::yaml-cpp)
set_target_properties(ray_cuboid_intersection
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(main
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

