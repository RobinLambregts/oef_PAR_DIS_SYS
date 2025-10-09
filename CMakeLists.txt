cmake_minimum_required(VERSION 3.30)
project(PAR_DIS_SYS)

set(CMAKE_CXX_STANDARD 23)

include_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(PAR_DIS_SYS oefensessie3/oef_9/main.cpp)

target_link_libraries(PAR_DIS_SYS msmpi)

