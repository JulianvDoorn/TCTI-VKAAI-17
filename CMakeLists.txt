cmake_minimum_required(VERSION 3.0.0)
project(tcti-vkaai-17)

set(cxxflags ${CMAKE_CXX_FLAGS}
    -std=c++17
    -Wall
    -Wextra
    -pedantic-errors
    -Wfatal-errors
    -Wcast-align
    -Wmissing-declarations
    -Wredundant-decls
    -Wuninitialized
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -Wno-pessimizing-move
    -Wno-redundant-move
    -Weffc++
)

string(REGEX REPLACE ";" " " cxxflags "${cxxflags}")
set(CMAKE_CXX_FLAGS ${cxxflags})

include_directories(include/)

set(sources
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${sources})