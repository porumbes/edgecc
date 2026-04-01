cmake_minimum_required(VERSION 3.15)
project(edgecc C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# Will add Flex/Bison targets in later tasks. For now, just build main.
add_executable(edgecc src/main.c src/ast.c)
target_include_directories(edgecc PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src src)

enable_testing()
