cmake_minimum_required(VERSION 3.15)
project(TestInCacheBlock LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/home/leonli/MGARD/install-cuda-hooper" CACHE STRING "MGARD prefix for find_package")

find_package(mgard CONFIG REQUIRED)

add_executable(TestInCacheBlock src/TestInCacheBlock.cpp)

target_include_directories(TestInCacheBlock PRIVATE
  "${CMAKE_PREFIX_PATH}/include"
)

target_link_libraries(TestInCacheBlock PUBLIC mgard::mgard)