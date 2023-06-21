cmake_minimum_required(VERSION 3.4)

project(uYoloAppLime LANGUAGES CXX)
set(CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "./ncnn;${CMAKE_PREFIX_PATH}")
#添加资源文件，非必须，一旦采用，注意修改相应的qrc文件名
#set(RCC_FILES rcc.qrc)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
#设置工程包含当前目录，非必须
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(NCNN_LIBRARIES ${PROJECT_SOURCE_DIR}/ncnn/lib/libncnn.a)
set(NCNN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ncnn/include)

#打开全局uic，非必须
set(CMAKE_AUTOUIC ON)
#打开全局moc,设置自动生成moc文件，一定要设置
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
#打开全局rcc，非必须，如需打开，注意修改qrc文件名
#set(CMAKE_AUTORCC ON)
#启用多核心编程
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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
        ${PROJECT_SOURCE_DIR}/source/
        ${NCNN_INCLUDE_DIRS}
)

#查找需要的Qt库文件，最好每一个库都要写，Qt也会根据依赖关系自动添加
find_package(QT NAMES Qt5 COMPONENTS Core REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Gui REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ncnn REQUIRED)

#查找当前文件夹中的所有源代码文件，也可以通过Set命令将所有文件设置为一个变量
FILE(GLOB SRC_FILES "source/*.cpp")
#查找设置当前文件夹中所有的头文件
FILE(GLOB HEAD_FILES "source/*.hpp")
#查找设置当前文件夹中所有的ui文件
FILE(GLOB UI_FILES "source/*.ui")

file(COPY models/ DESTINATION .)

file(GLOB_RECURSE uYoloAppLime_SOURCES "source/*.*")
#通过Ui文件生成对应的头文件，一定要添加
qt5_wrap_ui(WRAP_FILES ${UI_FILES})

#创建工程文件
add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEAD_FILES} ${RCC_FILES} ${WRAP_FILES} ${uYoloAppLime_SOURCES})

#添加Qt5依赖项
target_link_libraries(${PROJECT_NAME} PUBLIC Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${NCNN_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

