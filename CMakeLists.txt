cmake_minimum_required(VERSION 2.8.3)
project(Task)
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message("No appropriate opencv package found.")
endif()

include_directories(
    .
    /usr/local/include
    ${OpenCV_INCLUDE_DIRS}``
)

link_directories(
    /usr/local/lib
)

aux_source_directory(
    ./src SRC
)

add_executable(Task StereoMatching.cc
    ${SRC}
)

target_link_libraries(Task
    ${OpenCV_LIBS}
)