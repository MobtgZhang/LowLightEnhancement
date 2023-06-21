cmake_minimum_required(VERSION 3.4)

project(uYoloAppLime LANGUAGES CXX)
set(CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -pthread")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
#设置工程包含当前目录，非必须
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# 设置编译选项以针对 ARM 架构进行优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
endif()

#添加编译器选项
set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS -mfpu=neon)

#库路径
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${FFTW3_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/source/
)

find_package(OpenCV REQUIRED)
# find_package(OpenMP REQUIRED)

#查找当前文件夹中的所有源代码文件，也可以通过Set命令将所有文件设置为一个变量
FILE(GLOB SRC_FILES "sources/*.cpp")
#查找设置当前文件夹中所有的头文件
FILE(GLOB HEAD_FILES "sources/*.hpp")

#创建工程文件
add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEAD_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

