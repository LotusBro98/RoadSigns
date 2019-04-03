cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)

project( detect )
find_package( OpenCV REQUIRED PATHS /usr/share/ NO_DEFAULT_PATH)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( detect src/detect.cpp )
target_link_libraries( detect ${OpenCV_LIBS} )
