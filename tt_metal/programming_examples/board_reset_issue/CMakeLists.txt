cmake_minimum_required(VERSION 3.22...3.30)
project(board_reset_issue)

add_executable(board_reset_issue)
target_sources(board_reset_issue PRIVATE host.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(board_reset_issue PUBLIC TT::Metalium)