cmake_minimum_required(VERSION 3.15)
project(edgecc C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# Ensure build/src/ exists for Bison/Flex output files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)

BISON_TARGET(parser src/parser.y ${CMAKE_CURRENT_BINARY_DIR}/src/parser.tab.c
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/parser.tab.h)
FLEX_TARGET(lexer src/lexer.l ${CMAKE_CURRENT_BINARY_DIR}/src/lexer.yy.c)
ADD_FLEX_BISON_DEPENDENCY(lexer parser)

set(EDGECC_SOURCES
    src/main.c
    src/ast.c
    src/semantic.c
    src/ir.c
    src/codegen.c
    ${BISON_parser_OUTPUTS}
    ${FLEX_lexer_OUTPUTS}
)

add_executable(edgecc ${EDGECC_SOURCES})
target_include_directories(edgecc PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/src)

# Tests
enable_testing()

add_executable(test_lexer
    tests/unit/test_lexer.c
    src/ast.c
    ${BISON_parser_OUTPUTS}
    ${FLEX_lexer_OUTPUTS}
)
target_include_directories(test_lexer PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/src tests/unit)
add_test(NAME test_lexer COMMAND test_lexer)

add_executable(test_parser
    tests/unit/test_parser.c
    src/ast.c
    ${BISON_parser_OUTPUTS}
    ${FLEX_lexer_OUTPUTS}
)
target_include_directories(test_parser PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/src tests/unit)
add_test(NAME test_parser COMMAND test_parser)

add_executable(test_semantic
    tests/unit/test_semantic.c
    src/ast.c
    src/semantic.c
    ${BISON_parser_OUTPUTS}
    ${FLEX_lexer_OUTPUTS}
)
target_include_directories(test_semantic PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/src tests/unit)
add_test(NAME test_semantic COMMAND test_semantic)

add_executable(test_ir
    tests/unit/test_ir.c
    src/ast.c
    src/semantic.c
    src/ir.c
    ${BISON_parser_OUTPUTS}
    ${FLEX_lexer_OUTPUTS}
)
target_include_directories(test_ir PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/src tests/unit)
add_test(NAME test_ir COMMAND test_ir)

add_executable(test_codegen
    tests/unit/test_codegen.c
    src/ast.c
    src/semantic.c
    src/ir.c
    src/codegen.c
    ${BISON_parser_OUTPUTS}
    ${FLEX_lexer_OUTPUTS}
)
target_include_directories(test_codegen PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/src tests/unit)
add_test(NAME test_codegen COMMAND test_codegen)

add_test(
    NAME e2e_tests
    COMMAND ${CMAKE_COMMAND} -E env python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/run_e2e.py
        $<TARGET_FILE:edgecc>
)

add_test(
    NAME negative_tests
    COMMAND ${CMAKE_COMMAND} -E env python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/negative/run_negative.py
        $<TARGET_FILE:edgecc>
)

add_test(
    NAME e2e_iterative
    COMMAND ${CMAKE_COMMAND} -E env python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/iterative/run_iterative.py
        --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        --tests-dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/iterative
)
